"""Requirement ontology: extract, categorize, and build cross-reference graph."""

import re
from dataclasses import dataclass, field

from pypdf import PdfReader

from embedder import embed_texts

# IAEA SSR-2/1 chapter-based categories
CATEGORIES = {
    "Management": list(range(1, 4)),
    "Safety Fundamentals": list(range(4, 13)),
    "Design Basis": list(range(13, 21)),
    "Design Principles": list(range(21, 42)),
    "Reactor & Coolant": list(range(43, 54)),
    "Containment": list(range(54, 59)),
    "I&C Systems": list(range(59, 68)),
    "Support Systems": list(range(68, 78)),
    "Waste & Radiation": list(range(78, 83)),
}

# Domain-knowledge cross-references (key known relationships)
DOMAIN_EDGES = [
    # Defence in depth connects to many
    (7, 4, "Defence in depth implements fundamental safety functions"),
    (7, 13, "Defence in depth defines plant state categories"),
    (7, 19, "Defence in depth Level 3: design basis accidents"),
    (7, 20, "Defence in depth Level 4: design extension conditions"),
    (7, 42, "Defence in depth verified through safety analysis"),
    # Design basis chain
    (13, 14, "Plant states define design basis"),
    (14, 15, "Design basis determines design limits"),
    (14, 16, "Design basis derived from initiating events"),
    (16, 17, "Initiating events include internal/external hazards"),
    (16, 19, "Initiating events lead to design basis accidents"),
    (19, 20, "DBA extended to design extension conditions"),
    # Safety classification chain
    (22, 21, "Classification determines separation requirements"),
    (22, 23, "Classification determines reliability requirements"),
    (22, 25, "Safety class determines single failure criterion applicability"),
    # Reactor systems
    (43, 44, "Fuel performance requires structural capability"),
    (45, 46, "Core control includes reactor shutdown"),
    (47, 48, "Coolant system includes overpressure protection"),
    (47, 49, "Coolant system manages inventory"),
    (47, 50, "Coolant system includes cleanup"),
    (51, 52, "Residual heat removal includes emergency cooling"),
    (52, 53, "Emergency cooling transfers heat to ultimate heat sink"),
    # Containment chain
    (54, 55, "Containment controls radioactive releases"),
    (54, 56, "Containment requires isolation capability"),
    (54, 57, "Containment manages access"),
    (54, 58, "Containment conditions must be controlled"),
    # I&C chain
    (59, 60, "Instrumentation feeds control systems"),
    (60, 61, "Control systems include protection system"),
    (61, 62, "Protection system requires reliability/testability"),
    (65, 66, "Main control room backed by supplementary"),
    (66, 67, "Supplementary control room supports emergency centre"),
    # Support → Safety
    (68, 46, "Emergency power supports reactor shutdown"),
    (68, 52, "Emergency power supports emergency cooling"),
    (27, 68, "Support services include emergency power"),
    # Radiation protection
    (5, 81, "Radiation protection principle implemented in design"),
    (81, 82, "Radiation protection requires monitoring means"),
    # Cross-cutting
    (10, 42, "Safety assessment includes safety analysis"),
    (20, 54, "DEC includes containment integrity for severe accidents"),
    (24, 25, "Common cause failures complement single failure criterion"),
    (30, 31, "Qualification includes ageing management"),
    (32, 65, "Operator performance implemented in control room design"),
]


@dataclass
class RequirementNode:
    id: int
    title: str
    category: str
    page: int
    text_preview: str = ""


@dataclass
class RequirementEdge:
    source: int
    target: int
    relation: str
    edge_type: str = "domain"  # "domain" or "semantic"


def _get_category(req_id: int) -> str:
    for cat, ids in CATEGORIES.items():
        if req_id in ids:
            return cat
    return "Other"


def extract_requirements(pdf_path: str) -> list[RequirementNode]:
    """Extract all requirements from IAEA SSR-2/1 PDF."""
    reader = PdfReader(pdf_path)
    nodes: dict[int, RequirementNode] = {}

    for i, page in enumerate(reader.pages):
        pn = i + 1
        if pn < 20 or pn > 78:
            continue
        text = page.extract_text() or ""

        for m in re.finditer(
            r"Requirement\s+(\d+)\s*:\s*(.+?)(?=Requirement\s+\d+\s*:|$)",
            text,
            re.DOTALL,
        ):
            num = int(m.group(1))
            content = m.group(2)
            title = content.split("\n")[0].strip()
            preview = " ".join(content.split()[:60])

            if num not in nodes or len(content) > len(nodes[num].text_preview):
                nodes[num] = RequirementNode(
                    id=num,
                    title=title,
                    category=_get_category(num),
                    page=pn,
                    text_preview=preview[:300],
                )

    return sorted(nodes.values(), key=lambda n: n.id)


def build_semantic_edges(
    nodes: list[RequirementNode], threshold: float = 0.75
) -> list[RequirementEdge]:
    """Find semantically similar requirements using embedding cosine similarity."""
    texts = [f"Requirement {n.id}: {n.title}. {n.text_preview}" for n in nodes]
    embeddings = embed_texts(texts)

    edges = []
    import numpy as np

    emb_array = np.array(embeddings)
    sim_matrix = emb_array @ emb_array.T

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if nodes[i].category == nodes[j].category:
                continue  # skip same-category (already obvious)
            if sim_matrix[i][j] > threshold:
                edges.append(
                    RequirementEdge(
                        source=nodes[i].id,
                        target=nodes[j].id,
                        relation=f"Semantic similarity ({sim_matrix[i][j]:.2f})",
                        edge_type="semantic",
                    )
                )
    return edges


def build_graph(pdf_path: str) -> dict:
    """Build complete requirement cross-reference graph."""
    nodes = extract_requirements(pdf_path)

    # Domain knowledge edges
    domain_edges = [
        RequirementEdge(source=s, target=t, relation=r, edge_type="domain")
        for s, t, r in DOMAIN_EDGES
    ]

    # Semantic similarity edges
    semantic_edges = build_semantic_edges(nodes)

    return {
        "nodes": [
            {
                "id": n.id,
                "title": n.title,
                "category": n.category,
                "page": n.page,
            }
            for n in nodes
        ],
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "relation": e.relation,
                "type": e.edge_type,
            }
            for e in domain_edges + semantic_edges
        ],
        "categories": list(CATEGORIES.keys()),
        "stats": {
            "total_requirements": len(nodes),
            "domain_edges": len(domain_edges),
            "semantic_edges": len(semantic_edges),
        },
    }
