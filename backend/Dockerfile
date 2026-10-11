# Fallback host for the backend only (see docs/BUILD_GUIDE.md, Appendix 2).
FROM python:3.12-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PORT=7860
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt uvicorn[standard]
COPY --chown=user . .
EXPOSE 7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
