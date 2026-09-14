# Base Vendaval — painel público

Correção da rajada máxima de vento do ERA5 usando as estações do INMET.

Este painel cobre a **trilha de interpolação**: o dado observado, o mecanismo de correção
por resíduo, as versões V1–V5, o teto estrutural da combinação convexa e a comparação
leave-one-station-out entre os métodos. A trilha de IA é um trabalho separado e não está aqui.

## Percurso

| Grupo | Seções |
| --- | --- |
| A análise | O dado observado · O mecanismo e as versões · O teto estrutural |
| As evidências | A régua (LOOCV) · Os mapas · No tempo |
| O fecho | Para onde foi |

As métricas são calculadas no carregamento, a partir dos dados em `dados/` — não há número
digitado à mão.

## Estrutura

```
app/streamlit_app.py   painel
figuras/               saídas do pipeline
dados/                 observação, métricas LOOCV, campos espaciais e séries
```

## Rodar localmente

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app/streamlit_app.py
```

## Dados e código do pipeline

Este repositório guarda **apenas** os resultados publicados. Os dados do projeto ficam em
`/home/publico/vendaval/` na máquina 3 do cluster de pesquisa — o catálogo por arquivo está
em `docs/MAPA_DADOS.md` e o mapa de pastas em `docs/MAPA_DIRETORIOS.md`, ambos dentro daquele
diretório. O código do pipeline vive no repositório `Base Vendaval`, no Azure DevOps.
