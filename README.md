# Vendaval — painel público

Correção de viés da rajada máxima de vento: reanálise ERA5 calibrada contra as
estações automáticas do INMET.

O painel percorre a análise em seis seções — o problema, os dados, as covariáveis,
os modelos, a espacialização e o resultado — cada uma respondendo a uma pergunta.
As métricas de erro são calculadas na hora, a partir dos CSVs de resultado. O
acervo completo de figuras fica numa aba à parte.

## Estrutura

```
app/streamlit_app.py   painel
figuras/               saídas do pipeline, por etapa
dados/                 CSVs de resultado consultados pelo painel
```

## Rodar localmente

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app/streamlit_app.py
```

## Dados e código do pipeline

Este repositório guarda **apenas** os resultados publicados. Os dados do projeto
ficam em `/home/publico/vendaval/` na máquina 3 do cluster de pesquisa — o
catálogo por arquivo está em `docs/MAPA_DADOS.md` e o mapa de pastas em
`docs/MAPA_DIRETORIOS.md`, ambos dentro daquele diretório. O código do pipeline
vive no repositório `Base Vendaval`, no Azure DevOps.
