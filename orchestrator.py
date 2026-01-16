# orchestrator.py
# Central class to orchestrate components in MetaBlocks style

class CognitiveApp:
    def __init__(self):
        self.components = {}

    def with_component(self, name: str, component_class: type):
        """Add any component (pluggable from any tech stack)"""
        self.components[name] = component_class()
        return self  # Chainable

    def with_ai(self):
        from ai_core import AI_Core
        return self.with_component("ai", AI_Core)

    def with_translator(self):
        from translator import Translator
        return self.with_component("translator", Translator)

    def with_voice(self):
        from voice_block import Voice_Engine
        return self.with_component("voice", Voice_Engine)

    def with_logger(self):
        from logger import AppLogger
        return self.with_component("logger", AppLogger)

    def with_db(self):
        from db_block import DBBlock
        return self.with_component("db", DBBlock)

    def with_knowledge_graph(self):
        from services.blocks.knowledge_graph_v2 import KnowledgeUniverse
        return self.with_component("kg", KnowledgeUniverse)

    def with_personal_rag(self, user_id: str):
        from services.blocks.personal_rag_system import PersonalRAG
        from db_block import DBBlock  # Need supabase client
        supabase = DBBlock()  # Assuming DBBlock provides client
        self.components["rag"] = PersonalRAG(supabase.client, user_id)
        return self

    # Add more .with_xxx() for other components as needed (e.g., .with_auth())

    def build(self):
        """Auto-init and connect components if needed"""
        # Example: Connect logger to ai if exists
        if "logger" in self.components and "ai" in self.components:
            self.components["ai"].logger = self.components["logger"]
        # Add auto-optimize or self-heal logic here in future
        return self

    def run(self, mode: str, **kwargs):
        """Run specific module with components"""
        if mode == "weaver":
            import module_weaver as mw
            mw.run(self.components, **kwargs)
        elif mode == "cfo":
            import module_cfo as mc
            mc.run(self.components, **kwargs)
        # Add hot_swap or discover_components() in future for dynamic
