from llama_index.core.workflow import ( 
    Context, 
    Workflow
)

class ContextService:
    """Centralized service to manage the shared Context."""
    _instance = None

    @staticmethod
    def initialize(workflow: Workflow) -> None:
        """Initialize the global Context instance."""
        #if not ContextService._instance:
        ContextService._instance = Context(workflow)

    @staticmethod
    def get_context() -> Context:
        """Return the shared Context instance."""
        """ if not ContextService._instance:
            raise ValueError("ContextService has not been initialized.") """
        return ContextService._instance

    @staticmethod
    def set_context(ctx: Context) -> None:
        """Set the shared Context instance."""
        ContextService._instance = ctx

# Create a shared instance for import
context_service = ContextService()
