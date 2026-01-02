# KNOWLEDGE GRAPH V2 - Hệ thống Tri thức Đa tầng
# Triết lý: Dựa trên "The Order of Things" (Foucault) và "Thinking in Systems" (Meadows)
# Phiên bản cập nhật: Thêm sách tinh hoa để bao trùm 4 tầng triết học

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import streamlit as st
from datetime import datetime

class KnowledgeUniverse:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.encoder = self._load_encoder()
        self.episteme_layers = {
            "Toán học & Logic": [],
            "Vật lý & Sinh học": [],
            "Văn hóa & Quyền lực": [],
            "Ý thức & Giải phóng": []
        }

    @st.cache_resource
    def _load_encoder(_self):
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device='cpu')

    def add_book(self, title, content_summary, first_principles="", metadata=None):
        if metadata is None:
            metadata = {}
        node_id = f"book_{len(self.graph.nodes)}"
        embedding = self.encoder.encode([content_summary])[0]
        self.graph.add_node(node_id, type="book", title=title, embedding=embedding, 
                            added_at=datetime.now().isoformat(), first_principles=first_principles, **metadata)
        layer = self._classify_episteme(content_summary, metadata.get("tags", []))
        if layer in self.episteme_layers:
            self.episteme_layers[layer].append(node_id)
        self._auto_link_node(node_id)
        return node_id

    def _classify_episteme(self, text, tags):
        keywords_map = {
            "Toán học & Logic": ["logic", "math", "proof", "toán", "xác suất"],
            "Vật lý & Sinh học": ["physics", "evolution", "brain", "não bộ", "vật lý"],
            "Văn hóa & Quyền lực": ["power", "culture", "society", "quyền lực", "văn hóa"],
            "Ý thức & Giải phóng": ["consciousness", "mindfulness", "thiền", "ý thức"]
        }
        text_lower = text.lower()
        for layer, keywords in keywords_map.items():
            if any(kw in text_lower or kw in tags for kw in keywords):
                return layer
        return "Văn hóa & Quyền lực"

    def _auto_link_node(self, node_id, threshold=0.6):
        new_node = self.graph.nodes[node_id]
        new_emb = new_node["embedding"]
        new_time = datetime.fromisoformat(new_node["added_at"])
        for other_id in self.graph.nodes:
            if other_id == node_id:
                continue
            other_node = self.graph.nodes[other_id]
            other_emb = other_node["embedding"]
            other_time = datetime.fromisoformat(other_node["added_at"])
            sim = cosine_similarity([new_emb], [other_emb])[0][0]
            if sim > threshold:
                if new_time > other_time:
                    self.graph.add_edge(other_id, node_id, relation="influence", weight=sim, confidence=sim)
                else:
                    self.graph.add_edge(node_id, other_id, relation="reference", weight=sim, confidence=sim)

    def find_related_books(self, query_text, top_k=5):
        query_emb = self.encoder.encode([query_text])[0]
        results = []
        for node_id in self.graph.nodes:
            node = self.graph.nodes[node_id]
            if node["type"] != "book":
                continue
            sim = cosine_similarity([query_emb], [node["embedding"]])[0][0]
            path_explanation = self._explain_connection(query_text, node_id)
            results.append((node_id, node["title"], float(sim), path_explanation))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def _explain_connection(self, query, node_id):
        node = self.graph.nodes[node_id]
        layer = None
        for l, nodes in self.episteme_layers.items():
            if node_id in nodes:
                layer = l
                break
        neighbors = list(self.graph.neighbors(node_id))
        explanation = f"Thuộc tầng '{layer}'"
        if neighbors:
            neighbor_titles = [self.graph.nodes[n]["title"] for n in neighbors[:2]]
            explanation += f" | Liên quan: {', '.join(neighbor_titles)}"
        return explanation

    def get_episteme_summary(self):
        summary = {}
        for layer, node_ids in self.episteme_layers.items():
            books = [self.graph.nodes[nid]["title"] for nid in node_ids[-3:]]
            summary[layer] = {"count": len(node_ids), "recent": books}
        return summary

    def detect_contradictions(self, threshold=0.8):
        contradictions = []
        return contradictions

    def export_for_visualization(self):
        nodes = []
        edges = []
        color_map = {
            "Toán học & Logic": "#FF6B6B",
            "Vật lý & Sinh học": "#4ECDC4",
            "Văn hóa & Quyền lực": "#FFD93D",
            "Ý thức & Giải phóng": "#A8E6CF"
        }
        for node_id in self.graph.nodes:
            node_data = self.graph.nodes[node_id]
            layer = None
            for l, nids in self.episteme_layers.items():
                if node_id in nids:
                    layer = l
                    break
            nodes.append({"id": node_id, "label": node_data["title"], "color": color_map.get(layer, "#CCCCCC"), "size": 20})
        for u, v, data in self.graph.edges(data=True):
            edges.append({"source": u, "target": v, "label": data.get("relation", ""), "color": "#888888", "width": data.get("weight", 1) * 3})
        return nodes, edges

@st.cache_resource
def init_knowledge_universe():
    """Khởi tạo Knowledge Universe với 18 sách tinh hoa"""
    try:
        kg = KnowledgeUniverse()
        
        # ✅ SỬA: Bọc trong try-except để bắt lỗi cụ thể
        try:
            add_selected_books(kg)
        except Exception as e:
            st.warning(f"⚠️ Không thể thêm sách tinh hoa: {e}")
            # Vẫn trả về KG rỗng thay vì None
        
        return kg
        
    except Exception as e:
        st.error(f"❌ Lỗi khởi tạo KnowledgeUniverse: {e}")
        return None

def add_selected_books(kg: KnowledgeUniverse):
    """Thêm 18 sách tinh hoa vào Knowledge Graph - Fix biến selected_books"""
    
    # ✅ SỬA: Định nghĩa đúng tên biến
    selected_books = [
        # Tầng I: Toán học & Logic (4 sách)
        {
            "title": "Probability Theory: The Logic of Science",
            "author": "E.T. Jaynes",
            "summary": "Xác suất là logic suy luận từ dữ liệu, không phải tần suất. Mọi suy luận đều là cập nhật niềm tin dựa trên bằng chứng mới.",
            "first_principles": "Xác suất là mức độ tin tưởng hợp lý, không phải tần suất khách quan. Cập nhật niềm tin qua định lý Bayes: P(H|E) ∝ P(E|H) × P(H).",
            "tags": ["logic", "xác suất", "toán học"]
        },
        {
            "title": "Gödel, Escher, Bach",
            "author": "Douglas Hofstadter",
            "summary": "Ý thức xuất hiện từ vòng lặp tự tham chiếu (strange loop) trong hệ thống hình thức.",
            "first_principles": "Hệ thống đủ phức tạp tạo ý nghĩa từ tự lặp. Logic có giới hạn nội tại (định lý Gödel).",
            "tags": ["logic", "math", "ý thức"]
        },
        {
            "title": "Thinking Fast and Slow",
            "author": "Daniel Kahneman",
            "summary": "Hai hệ thống tư duy: System 1 (nhanh, trực giác, thiên kiến) và System 2 (chậm, phản biện, logic).",
            "first_principles": "Trực giác thường sai lệch. Phải dùng tư duy chậm để phát hiện và sửa lỗi logic.",
            "tags": ["logic", "proof", "nhận thức"]
        },
        {
            "title": "Fooled by Randomness",
            "author": "Nassim Nicholas Taleb",
            "summary": "Con người thường nhầm lẫn giữa kỹ năng và may mắn. Ngẫu nhiên chi phối cuộc sống nhiều hơn ta nghĩ.",
            "first_principles": "Tập trung vào bền vững (antifragile), không vào kết quả ngắn hạn. Tránh ảo tưởng kiểm soát.",
            "tags": ["xác suất", "logic", "rủi ro"]
        },
        
        # Tầng II: Vật lý & Sinh học (5 sách)
        {
            "title": "Order out of Chaos",
            "author": "Ilya Prigogine",
            "summary": "Hỗn loạn không phải kẻ thù của trật tự. Hệ thống xa cân bằng tự tổ chức tạo cấu trúc mới qua entropy.",
            "first_principles": "Thời gian bất đối xứng (không thể đảo ngược). Hệ phức hợp tự tổ chức xa điểm cân bằng nhiệt động.",
            "tags": ["physics", "evolution", "hệ thống"]
        },
        {
            "title": "The Selfish Gene",
            "author": "Richard Dawkins",
            "summary": "Gen, không phải cá thể hay loài, là đơn vị chọn lọc tự nhiên. Sinh vật là 'máy sống' phục vụ sao chép gen.",
            "first_principles": "Hành vi lợi ích vì lợi ích gen. Tiến hóa là gene-centric, không group-centric.",
            "tags": ["evolution", "brain", "sinh học"]
        },
        {
            "title": "Thinking in Systems",
            "author": "Donella Meadows",
            "summary": "Thế giới là tập hợp các hệ thống với vòng phản hồi. Hiểu hệ thống mới can thiệp hiệu quả.",
            "first_principles": "Tìm điểm đòn bẩy (leverage point) để thay đổi hệ thống từ gốc. Feedback loop chi phối hành vi.",
            "tags": ["physics", "evolution", "hệ thống"]
        },
        {
            "title": "Antifragile",
            "author": "Nassim Nicholas Taleb",
            "summary": "Có hệ thống không chỉ chống chịu được hỗn loạn mà còn cải thiện từ nó (antifragile).",
            "first_principles": "Lợi ích từ biến động và stress. Thử và sai là cách học của hệ phức hợp.",
            "tags": ["physics", "evolution", "rủi ro"]
        },
        {
            "title": "Behave",
            "author": "Robert Sapolsky",
            "summary": "Hành vi con người là kết quả đa tầng: từ hormone (giây), não bộ (phút), gen (triệu năm), đến văn hóa.",
            "first_principles": "Hành vi không có nguyên nhân đơn. Phải phân tích đa tầng thời gian và không gian.",
            "tags": ["brain", "evolution", "sinh học"]
        },
        
        # Tầng III: Văn hóa & Quyền lực (4 sách)
        {
            "title": "Leviathan",
            "author": "Thomas Hobbes",
            "summary": "Trong trạng thái tự nhiên, con người ở 'chiến tranh của mọi người chống lại mọi người'. Cần hợp đồng xã hội.",
            "first_principles": "Bản chất con người là tự bảo tồn. Quyền lực tuyệt đối cần thiết cho hòa bình xã hội.",
            "tags": ["power", "society", "chính trị"]
        },
        {
            "title": "The Structure of Scientific Revolutions",
            "author": "Thomas Kuhn",
            "summary": "Khoa học không tiến bộ tuyến tính. Thay đổi qua 'cách mạng paradigm' khi mô hình cũ sụp đổ.",
            "first_principles": "Mô hình khoa học (paradigm) thay đổi không tích lũy. Ngữ cảnh văn hóa chi phối chân lý.",
            "tags": ["culture", "power", "khoa học"]
        },
        {
            "title": "Sapiens",
            "author": "Yuval Noah Harari",
            "summary": "Homo sapiens thống trị nhờ khả năng hợp tác linh hoạt qua 'trật tự tưởng tượng' (tôn giáo, tiền, pháp luật).",
            "first_principles": "Huyền thoại chung cho phép hợp tác quy mô lớn. Văn hóa tạo thực tại xã hội.",
            "tags": ["culture", "society", "lịch sử"]
        },
        {
            "title": "The Dawn of Everything",
            "author": "David Graeber & David Wengrow",
            "summary": "Xã hội cổ đại đa dạng và linh hoạt hơn ta tưởng. Lịch sử không phải tiến hóa tuyến tính từ bình đẳng đến bất bình đẳng.",
            "first_principles": "Con người luôn thử nghiệm xã hội. Tự do là lựa chọn, không phải tất yếu lịch sử.",
            "tags": ["culture", "power", "lịch sử"]
        },
        
        # Tầng IV: Ý thức & Giải phóng (5 sách)
        {
            "title": "The Origin of Consciousness in the Breakdown of the Bicameral Mind",
            "author": "Julian Jaynes",
            "summary": "Ý thức (self-awareness) xuất hiện cách đây 3000 năm khi tâm trí 'nhị phân' (nghe giọng thần) sụp đổ.",
            "first_principles": "Ý thức là tường thuật nội tại về bản thân. Ý thức là sản phẩm văn hóa, không phải sinh học thuần túy.",
            "tags": ["consciousness", "mindfulness", "tâm lý"]
        },
        {
            "title": "Phenomenology of Perception",
            "author": "Maurice Merleau-Ponty",
            "summary": "Nhận thức không tách rời cơ thể. Ý thức là 'cơ thể sống trong thế giới' (embodied mind).",
            "first_principles": "Cơ thể là trung tâm kinh nghiệm. Ý thức không phải Descartes' 'tâm trí tách biệt'.",
            "tags": ["consciousness", "thiền", "triết học"]
        },
        {
            "title": "The Way of Zen",
            "author": "Alan Watts",
            "summary": "Zen là trải nghiệm trực tiếp thực tại, vượt ngôn ngữ và nhị nguyên chủ-khách.",
            "first_principles": "Không tâm (mushin). Buông xả nỗ lực kiểm soát, sống tự nhiên (wu-wei).",
            "tags": ["mindfulness", "ý thức", "thiền"]
        },
        {
            "title": "Steps to an Ecology of Mind",
            "author": "Gregory Bateson",
            "summary": "Tâm trí không nằm trong đầu. Tâm trí là hệ thống sinh thái với vòng phản hồi (feedback loops).",
            "first_principles": "Học là thay đổi 'khung' (frame). Hệ thống tự điều chỉnh qua thông tin.",
            "tags": ["consciousness", "mindfulness", "hệ thống"]
        },
        {
            "title": "A History of Western Philosophy",
            "author": "Bertrand Russell",
            "summary": "Lịch sử triết học Tây phương từ tiền Socrates đến thế kỷ 20, với phê phán xã hội sắc bén.",
            "first_principles": "Triết học gắn với ngữ cảnh lịch sử. Logic và lý tính dẫn đường giải phóng con người.",
            "tags": ["consciousness", "ý thức", "triết học"]
        }
    ]
    
    # ✅ SỬA: Thêm sách vào KG một cách an toàn
    success_count = 0
    for book in selected_books:
        try:
            metadata = {
                "author": book["author"],
                "tags": book["tags"]
            }
            kg.add_book(
                title=book["title"],
                content_summary=book["summary"],
                first_principles=book["first_principles"],
                metadata=metadata
            )
            success_count += 1
        except Exception as e:
            st.warning(f"⚠️ Không thêm được '{book['title']}': {e}")
            continue
    
    if success_count > 0:
        st.success(f"✅ Đã thêm {success_count}/{len(selected_books)} sách tinh hoa vào Knowledge Graph")
    else:
        st.error("❌ Không thêm được sách nào vào Knowledge Graph")
        
def upgrade_existing_database(excel_path, kg: KnowledgeUniverse):
    """Nâng cấp KG hiện có bằng cách thêm sách từ Excel"""
    import pandas as pd
    
    try:
        df = pd.read_excel(excel_path).dropna(subset=["Tên sách"])
        
        success_count = 0
        for idx, row in df.iterrows():
            try:
                title = str(row["Tên sách"]).strip()
                summary = str(row.get("CẢM NHẬN", "")).strip()
                
                # Bỏ qua sách không có summary
                if not summary or summary == "nan":
                    continue
                
                metadata = {
                    "author": str(row.get("Tác giả", "Unknown")),
                    "tags": [t.strip() for t in str(row.get("Tags", "")).split(",") if t.strip()]
                }
                
                # Thêm vào KG (không có first_principles từ Excel)
                kg.add_book(title, summary, first_principles="", metadata=metadata)
                success_count += 1
                
            except Exception as e:
                continue  # Bỏ qua sách lỗi
        
        st.info(f"📚 Đã thêm {success_count}/{len(df)} sách từ Excel vào Knowledge Graph")
        return kg
        
    except Exception as e:
        st.error(f"❌ Lỗi đọc Excel: {e}")
        return kg
