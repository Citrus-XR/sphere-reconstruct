"""pipeline layer.

Engine + Stage 抽象 + レジストリ. Worker プロセス側で完結する.
FastAPI プロセスから触ってよいのは PipelineState / StageName / Manifest だけ.
"""
