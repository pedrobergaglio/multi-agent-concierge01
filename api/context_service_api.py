from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from llama_index.llms.openai import OpenAI

app = FastAPI()

# Shared context variable
shared_context = None

class ContextData(BaseModel):
    data: Dict[str, Any]

@app.get("/context/")
def get_context():
    """
    Retrieve the current shared Context.
    """
    global shared_context
    if not shared_context:
        return {"data": {}}
        #raise HTTPException(status_code=404, detail="Context not initialized")
    return {"data": shared_context}

@app.post("/context/")
def set_context(context_data: ContextData):
    """
    Update the shared Context.
    """
    global shared_context
    try:
        if not shared_context:
            raise HTTPException(status_code=404, detail="Context not initialized")
        
        # Deserialize objects
        def deserialize(obj):
            if isinstance(obj, dict) and obj.get("type") == "OpenAI":
                return OpenAI(model=obj["model"], temperature=obj["temperature"])
            return obj

        shared_context = {k: deserialize(v) for k, v in context_data.data.items()}
        return {"message": "Context updated"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to set context: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)