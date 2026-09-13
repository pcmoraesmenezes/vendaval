# Vendaval — painel público

Vitrine de resultados do projeto de correção de viés da rajada máxima de vento:
reanálise ERA5 calibrada contra as estações automáticas do INMET, com recorte de
aprofundamento no Cluster 3 (Sul do Brasil).

O painel reúne, num só lugar, as figuras já produzidas pelo pipeline e os dois
conjuntos de resultado que valem consulta interativa (comparação das versões
corrigidas por estação e ranking de preditores).

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
