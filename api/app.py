from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from pathlib import Path
import logging
from typing import List

from .schemas import (
    SearchRequest, SearchResponse, CodeResult,
    EmbedRequest, EmbedResponse, HealthResponse
)
from models.fusion_model import FusionModel
from inference.vector_store import VectorStore
from inference.search_engine import SearchEngine
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Code Search Engine API",
    description="CodeBERT + GNN powered code search engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

search_engine = None


@app.on_event("startup")
async def startup_event():
    """Initialize models and index on startup"""
    global search_engine
    
    logger.info("Loading model and index...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    model = FusionModel(
        codebert_model=config.get('model', 'codebert_model'),
        gnn_hidden_dim=config.get('model', 'gnn_hidden_dim'),
        gnn_num_layers=config.get('model', 'gnn_num_layers'),
        gnn_type=config.get('model', 'gnn_type'),
        gnn_heads=config.get('model', 'gnn_heads'),
        dropout=config.get('model', 'dropout'),
        fusion_method=config.get('model', 'fusion_method'),
        final_embedding_dim=config.get('model', 'final_embedding_dim')
    )
    
    checkpoint_path = Path(config.get('paths', 'model_save_dir')) / 'best_model.pt'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from {checkpoint_path}")
    else:
        logger.warning("No checkpoint found, using randomly initialized model")
    
    vector_store = VectorStore(
        embedding_dim=config.get('model', 'final_embedding_dim'),
        index_type=config.get('inference', 'faiss_index_type'),
        nlist=config.get('inference', 'faiss_nlist'),
        nprobe=config.get('inference', 'faiss_nprobe'),
        use_gpu=device == 'cuda'
    )
    
    index_path = Path(config.get('paths', 'index_save_dir'))
    if (index_path / 'faiss_index.bin').exists():
        vector_store.load(str(index_path))
        logger.info(f"Loaded index from {index_path}")
    else:
        logger.warning("No index found, starting with empty index")
    
    search_engine = SearchEngine(
        model=model,
        vector_store=vector_store,
        device=device
    )
    
    logger.info("Startup complete")


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=search_engine is not None,
        index_size=search_engine.vector_store.index.ntotal if search_engine else 0
    )


@app.post("/search", response_model=SearchResponse)
async def search_code(request: SearchRequest):
    """Search for code snippets"""
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    try:
        results = search_engine.search(
            query=request.query,
            top_k=request.top_k
        )
        
        if request.language:
            results = [r for r in results if r['language'] == request.language]
        
        code_results = [CodeResult(**r) for r in results]
        
        return SearchResponse(
            query=request.query,
            results=code_results,
            total_results=len(code_results)
        )
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed", response_model=EmbedResponse)
async def embed_code(request: EmbedRequest):
    """Get embedding for a code snippet"""
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    try:
        embedding = search_engine._encode_code_batch([
            {'code': request.code, 'language': request.language}
        ])
        
        embedding_list = embedding[0].cpu().tolist()
        
        return EmbedResponse(
            embedding=embedding_list,
            dimension=len(embedding_list)
        )
    
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index")
async def index_code(code_snippets: List[dict]):
    """Index new code snippets"""
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    try:
        search_engine.index_code(code_snippets)
        
        return {
            "status": "success",
            "indexed": len(code_snippets),
            "total_index_size": search_engine.vector_store.index.ntotal
        }
    
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)