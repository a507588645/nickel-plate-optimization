from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from starlette.concurrency import run_in_threadpool

from app_logging import clear_request_context, set_request_context
from app_metrics import HTTP_REQUEST_EXCEPTIONS_TOTAL, HTTP_REQUEST_LATENCY_SECONDS, HTTP_REQUESTS_TOTAL, render_metrics
from app_settings import get_app_settings
from inference import (
    ab_detect,
    detect_plate,
    dianjie_detect,
    dianji_detect,
    huaxian,
    is_nieban,
    nietiao_segment,
    preprocess_image,
)
from runtime_state import preload_models, runtime_state, shutdown_models


settings = get_app_settings()
logger = logging.getLogger("jinchuan.api")


# --------------------------- 请求 / 响应模型 ---------------------------


class ApiResponse(BaseModel):
    code: int
    msg: str
    data: dict


class PreprocessRequest(BaseModel):
    image: str
    out_imagedirpath: str


class AbDetectRequest(BaseModel):
    image: str
    value1: int
    value2: int
    value3: int


class ImageRequest(BaseModel):
    image: str


class CuttingRequest(BaseModel):
    image1: str
    image2: str
    json_file1: str
    json_file2: str


class DetectPlateJsonRequest(BaseModel):
    image: str
    surface: str | None = None
    plate_type: str | None = None
    out_dir: str | None = None


# --------------------------- 生命周期 ---------------------------


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        "service_starting",
        extra={
            "event": "service_starting",
            "mode": settings.runtime.mode,
            "port": settings.server.port,
            "workers": settings.server.workers,
        },
    )
    if settings.runtime.preload_models:
        preload_models(logger)
    yield
    shutdown_models(logger)


app = FastAPI(
    title="Jinchuan Python WebAPI",
    version="2.0.0",
    docs_url="/docs" if settings.server.docs_enabled else None,
    redoc_url="/redoc" if settings.server.docs_enabled else None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors.allow_origins,
    allow_methods=settings.cors.allow_methods,
    allow_headers=settings.cors.allow_headers,
)


# --------------------------- 中间件 ---------------------------


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get(settings.logging.request_id_header) or uuid.uuid4().hex
    set_request_context(request_id=request_id, method=request.method, path=request.url.path)
    start_ts = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed = time.perf_counter() - start_ts
        HTTP_REQUEST_EXCEPTIONS_TOTAL.labels(method=request.method, path=request.url.path).inc()
        HTTP_REQUEST_LATENCY_SECONDS.labels(method=request.method, path=request.url.path).observe(elapsed)
        logger.exception(
            "request_failed",
            extra={
                "event": "request_failed",
                "status_code": 500,
                "duration_ms": round(elapsed * 1000, 3),
            },
        )
        clear_request_context()
        raise

    elapsed = time.perf_counter() - start_ts
    response.headers[settings.logging.request_id_header] = request_id
    HTTP_REQUESTS_TOTAL.labels(
        method=request.method,
        path=request.url.path,
        status_code=str(response.status_code),
    ).inc()
    HTTP_REQUEST_LATENCY_SECONDS.labels(method=request.method, path=request.url.path).observe(elapsed)
    logger.info(
        "request_completed",
        extra={
            "event": "request_completed",
            "status_code": response.status_code,
            "duration_ms": round(elapsed * 1000, 3),
        },
    )
    clear_request_context()
    return response


# --------------------------- 统一异常处理 ---------------------------


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning("request_validation_failed", extra={"event": "request_validation_failed", "errors": exc.errors()})
    return JSONResponse(
        status_code=422,
        content={"code": 0, "msg": f"参数校验失败: {exc.errors()}", "data": {}},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_exception", extra={"event": "unhandled_exception"})
    return JSONResponse(
        status_code=500,
        content={"code": 0, "msg": repr(exc), "data": {}},
    )


# --------------------------- 管理接口 ---------------------------


@app.get(settings.management.health_path)
async def health() -> JSONResponse:
    payload = {
        "status": runtime_state.status,
        "ready": runtime_state.ready,
        "models_loaded": runtime_state.models_loaded,
        "device": runtime_state.device,
        "uptime_seconds": round(time.time() - runtime_state.started_at, 3),
        "last_error": runtime_state.last_error,
    }
    status_code = 200 if runtime_state.ready else 503
    return JSONResponse(payload, status_code=status_code)


@app.get(settings.management.config_path)
async def config() -> JSONResponse:
    payload = {
        "settings": settings.public_dict(),
        "models": runtime_state.models,
        "gpu_enabled": runtime_state.device == "cuda",
    }
    return JSONResponse(payload)


@app.get(settings.management.metrics_path)
async def metrics() -> Response:
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


# --------------------------- 业务接口 ---------------------------


@app.post("/preprocess", response_model=ApiResponse)
async def preprocess(req: PreprocessRequest) -> dict:
    return await run_in_threadpool(preprocess_image, req.image, req.out_imagedirpath)


@app.post("/abdetect", response_model=ApiResponse)
async def abdetect(req: AbDetectRequest) -> dict:
    return await run_in_threadpool(
        ab_detect, req.image, req.value1, req.value2, req.value3
    )


@app.post("/detect/plate", response_model=ApiResponse)
async def detect_plate_endpoint(
    request: Request,
    surface: str = "auto",
    plate_type: str = "dianjie",
    out_dir: str | None = None,
) -> dict | JSONResponse:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()

    if content_type == "application/octet-stream":
        image_bytes = await request.body()
        if not image_bytes:
            return JSONResponse(
                status_code=422,
                content={"code": 0, "msg": "请求体不能为空", "data": {}},
            )
        return await run_in_threadpool(
            detect_plate,
            image_bytes,
            None,
            surface,
            plate_type,
            out_dir,
        )

    if content_type in {"", "application/json"} or content_type.endswith("+json"):
        try:
            payload = await request.json()
            req = DetectPlateJsonRequest.model_validate(payload)
        except (ValueError, ValidationError) as exc:
            return JSONResponse(
                status_code=422,
                content={"code": 0, "msg": f"参数校验失败: {exc}", "data": {}},
            )

        return await run_in_threadpool(
            detect_plate,
            None,
            req.image,
            req.surface if req.surface is not None else surface,
            req.plate_type if req.plate_type is not None else plate_type,
            req.out_dir if req.out_dir is not None else out_dir,
        )

    return JSONResponse(
        status_code=415,
        content={"code": 0, "msg": "Content-Type 必须为 application/octet-stream 或 application/json", "data": {}},
    )


@app.post("/dianjiEclass", response_model=ApiResponse)
async def dianjie_detection(req: ImageRequest) -> dict:
    return await run_in_threadpool(dianjie_detect, req.image)


@app.post("/dianjiclass", response_model=ApiResponse)
async def dianji_detection(req: ImageRequest) -> dict:
    return await run_in_threadpool(dianji_detect, req.image)


@app.post("/cutting", response_model=ApiResponse)
async def cutting(req: CuttingRequest) -> dict:
    return await run_in_threadpool(
        huaxian, req.image1, req.image2, req.json_file1, req.json_file2
    )


@app.post("/nietiaoSeg", response_model=ApiResponse)
async def nietiao_seg(req: ImageRequest) -> dict:
    return await run_in_threadpool(nietiao_segment, req.image)


@app.post("/isNieban", response_model=ApiResponse)
async def is_nieban_endpoint(req: ImageRequest) -> dict:
    return await run_in_threadpool(is_nieban, req.image)
