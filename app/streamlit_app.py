"""Painel público do projeto Vendaval — vitrine de resultados e mapa de onde vivem os dados."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
FIGURAS = RAIZ / "figuras"
DADOS = RAIZ / "dados"

st.set_page_config(
    page_title="Vendaval — Correção de rajada de vento",
    page_icon="🌬️",
    layout="wide",
)


# ---------------------------------------------------------------- utilidades


@st.cache_data
def carregar_csv(nome: str) -> pd.DataFrame:
    return pd.read_csv(DADOS / nome)


def listar(pasta: Path, prefixo: str = "", sufixo: str = ".png") -> list[Path]:
    """Arquivos de uma pasta, em ordem alfabética, filtrados por prefixo."""
    if not pasta.is_dir():
        return []
    return sorted(p for p in pasta.iterdir() if p.name.startswith(prefixo) and p.suffix == sufixo)


def titulo_legivel(caminho: Path) -> str:
    return caminho.stem.replace("_", " ")


def galeria(arquivos: list[Path], colunas: int = 2) -> None:
    """Grade de imagens. Sem trava de proporção — evita o encolhimento de primeiro
    desenho que aparece quando um gráfico é montado dentro de aba ainda escondida."""
    if not arquivos:
        st.info("Nenhuma figura disponível nesta seção.")
        return
    for inicio in range(0, len(arquivos), colunas):
        linha = st.columns(colunas)
        for coluna, arquivo in zip(linha, arquivos[inicio : inicio + colunas]):
            with coluna:
                st.image(str(arquivo), caption=titulo_legivel(arquivo), width="stretch")


# ---------------------------------------------------------------- cabeçalho

st.title("🌬️ Vendaval — correção de rajada de vento")
st.caption(
    "Correção de viés da rajada máxima da reanálise ERA5 contra as estações "
    "automáticas do INMET, com recorte de aprofundamento no Cluster 3 (Sul do Brasil)."
)

abas = st.tabs(
    [
        "📌 Visão geral",
        "🗺️ Exploração",
        "📊 Baseline e comparação",
        "🎯 Correção de viés",
        "🏅 Ranking de preditores",
    ]
)


# ---------------------------------------------------------------- visão geral

with abas[0]:
    esquerda, direita = st.columns([3, 2])

    with esquerda:
        st.subheader("O que o projeto faz")
        st.markdown(
            """
A reanálise ERA5 cobre o país inteiro em grade regular, mas **subestima a rajada
máxima** — o extremo é justamente o que ela mais suaviza. As estações automáticas do
INMET medem a rajada de verdade, só que em pontos esparsos.

O projeto liga as duas pontas: pareia cada estação à célula de grade correspondente,
aprende o erro do ERA5 em função de covariáveis meteorológicas e devolve um campo de
rajada corrigido — com cobertura de grade e magnitude calibrada pela observação.

**Etapas do pipeline**

1. **Aquisição** — reanálise ERA5 (níveis de superfície e de pressão) e série histórica das estações do INMET.
2. **Pré-processamento** — recorte espacial, agregação horária → diária e engenharia de covariáveis.
3. **Pareamento** — cada estação ligada à célula de grade que a contém, formando o conjunto de treino.
4. **Modelos de correção** — regressão linear regularizada, floresta aleatória e regressão quantílica, avaliadas por estação.
5. **Análise** — comparação das versões corrigidas contra a observação e ranqueamento dos preditores.
            """
        )

    with direita:
        st.subheader("Onde estão os dados")
        st.markdown(
            """
| Onde | Caminho | O que tem |
| --- | --- | --- |
| Máquina 3 do cluster de pesquisa | `/home/publico/vendaval/` | Árvore completa do projeto na infraestrutura compartilhada: entradas, intermediários e saídas |
| Azure DevOps — projeto `Base Vendaval` | repositório Git | Código do pipeline, da aquisição à análise |
| Estação de trabalho local | — | Reanálise horária do Cluster 3 ainda em transferência; não replicada no cluster |
            """
        )
        st.markdown(
            """
Dentro de `/home/publico/vendaval/docs/` há dois documentos de navegação:

- `MAPA_DADOS.md` — uma linha por arquivo de dado: o que é, qual script escreveu e de qual dado ele nasceu.
- `MAPA_DIRETORIOS.md` — mapa das pastas do projeto.
            """
        )
        st.info(
            "As figuras deste painel são as saídas já geradas pelo pipeline. "
            "Os dados brutos não são publicados aqui — ficam nos caminhos acima.",
            icon="📁",
        )

    st.divider()
    contagens = {
        "Exploração": len(listar(FIGURAS / "01_exploracao")),
        "Baseline": len(listar(FIGURAS / "02_baseline")),
        "Comparação e ajuste": len(listar(FIGURAS / "03_comparacao")),
        "Modelos de correção": len(listar(FIGURAS / "04_correcao" / "regressao"))
        + len(listar(FIGURAS / "04_correcao" / "quantilica")),
        "Ranking de preditores": len(listar(FIGURAS / "05_predep")),
    }
    colunas = st.columns(len(contagens))
    for coluna, (rotulo, quantidade) in zip(colunas, contagens.items()):
        coluna.metric(rotulo, f"{quantidade} figuras")


# ---------------------------------------------------------------- exploração

with abas[1]:
    st.subheader("Exploração dos dados de entrada")
    st.caption(
        "Panorama do que entra no pipeline: onde estão as estações, como a rajada se "
        "distribui e como se comportam as covariáveis candidatas."
    )

    pasta = FIGURAS / "01_exploracao"
    grupos = {
        "Estações e rajada observada": ["inmet_stations_map", "map_station_max_gusts", "pdf_gusts_density"],
        "Distribuições": listar(pasta, "boxplot"),
        "Campos médios da reanálise": listar(pasta, "10m_") + listar(pasta, "2m_") + listar(pasta, "mean_")
        + listar(pasta, "surface_") + listar(pasta, "total_") + listar(pasta, "significant_"),
        "Covariáveis por estação": listar(pasta, "map_covariate"),
        "Recorte de interesse": listar(pasta, "map_roi"),
        "Correlação entre covariáveis": ["heatmap_correlation"],
    }

    for nome_grupo, itens in grupos.items():
        arquivos = [
            pasta / f"{item}.png" if isinstance(item, str) else item for item in itens
        ]
        arquivos = [a for a in arquivos if a.exists()]
        if not arquivos:
            continue
        with st.container():
            st.markdown(f"**{nome_grupo}** · {len(arquivos)} figura(s)")
            mostrar = st.toggle("mostrar", key=f"exp_{nome_grupo}", value=nome_grupo.startswith("Estações"))
            if mostrar:
                galeria(arquivos, colunas=3 if len(arquivos) > 4 else 2)
            st.divider()


# ---------------------------------------------------------------- baseline

with abas[2]:
    st.subheader("Baseline e comparação entre versões")
    st.caption(
        "Ponto de partida do modelo e comparação visual entre as versões do campo corrigido."
    )

    st.markdown("**Baseline**")
    galeria(listar(FIGURAS / "02_baseline"), colunas=3)

    st.divider()
    st.markdown("**Comparação entre versões e ajuste de hiperparâmetros**")
    galeria(listar(FIGURAS / "03_comparacao"), colunas=2)


# ---------------------------------------------------------------- correção

with abas[3]:
    st.subheader("Correção de viés — resultado por estação")

    dados = carregar_csv("scatter_p95_p99_dados.csv")
    versoes = {
        "ERA5 original": ("era_orig_p95", "era_orig_p99"),
        "Corrigido V2": ("v2_p95", "v2_p99"),
        "Corrigido V3": ("v3_p95", "v3_p99"),
    }

    st.markdown(
        "Cada ponto é uma estação do INMET. O eixo horizontal traz o percentil "
        "observado na estação; o vertical, o valor do campo na mesma posição. "
        "Quanto mais perto da diagonal, menor o viés."
    )

    escolha_percentil = st.radio(
        "Percentil da rajada", ["p95", "p99"], horizontal=True, key="corr_percentil"
    )
    indice = 0 if escolha_percentil == "p95" else 1
    coluna_inmet = f"inmet_{escolha_percentil}"

    figura = go.Figure()
    resumo = []
    for nome_versao, colunas_versao in versoes.items():
        coluna = colunas_versao[indice]
        recorte = dados[[coluna_inmet, coluna, "codigo_estacao"]].dropna()
        figura.add_trace(
            go.Scattergl(
                x=recorte[coluna_inmet],
                y=recorte[coluna],
                mode="markers",
                name=nome_versao,
                marker=dict(size=5, opacity=0.6),
                hovertemplate="%{customdata}<br>observado %{x:.2f} m/s<br>campo %{y:.2f} m/s<extra></extra>",
                customdata=recorte["codigo_estacao"],
            )
        )
        erro = recorte[coluna] - recorte[coluna_inmet]
        resumo.append(
            {
                "Versão": nome_versao,
                "Viés médio (m/s)": round(erro.mean(), 3),
                "Erro absoluto médio (m/s)": round(erro.abs().mean(), 3),
                "Raiz do erro quadrático (m/s)": round((erro**2).mean() ** 0.5, 3),
                "Estações": len(recorte),
            }
        )

    limite_min = float(min(dados[coluna_inmet].min(), dados[[c[indice] for c in versoes.values()]].min().min()))
    limite_max = float(max(dados[coluna_inmet].max(), dados[[c[indice] for c in versoes.values()]].max().max()))
    figura.add_trace(
        go.Scatter(
            x=[limite_min, limite_max],
            y=[limite_min, limite_max],
            mode="lines",
            name="acerto perfeito",
            line=dict(dash="dash", width=1, color="#888888"),
            hoverinfo="skip",
        )
    )
    figura.update_layout(
        height=560,
        xaxis_title=f"Rajada observada — {escolha_percentil} (m/s)",
        yaxis_title=f"Rajada do campo — {escolha_percentil} (m/s)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    figura.update_xaxes(range=[limite_min, limite_max])
    figura.update_yaxes(range=[limite_min, limite_max])
    st.plotly_chart(figura, width="stretch", config={"responsive": True})

    st.markdown("**Erro contra a observação**, calculado sobre as estações do gráfico acima:")
    st.dataframe(pd.DataFrame(resumo), hide_index=True, width="stretch")
    st.caption(
        "Viés negativo significa campo abaixo da estação. Os números são de percentis "
        "por estação, não de erro instantâneo."
    )

    st.divider()
    st.markdown("**Avaliação dos modelos, estação a estação**")

    familias = {
        "Regressão linear e floresta aleatória": FIGURAS / "04_correcao" / "regressao",
        "Regressão quantílica": FIGURAS / "04_correcao" / "quantilica",
    }
    escolha_familia = st.selectbox("Família de modelo", list(familias), key="corr_familia")
    pasta_modelos = familias[escolha_familia]
    arquivos_modelos = listar(pasta_modelos)

    estacoes = sorted({a.stem.split("_")[-1] for a in arquivos_modelos})
    tipos = {
        "importance": "Importância das variáveis",
        "residuals": "Resíduos",
        "scatter": "Previsto × observado",
        "multi": "Múltiplos quantis",
    }
    # Só oferece o diagnóstico que a família selecionada realmente produziu —
    # "múltiplos quantis" não existe fora da regressão quantílica, e um seletor
    # que leva a uma seção vazia parece defeito.
    tipos = {
        prefixo: rotulo
        for prefixo, rotulo in tipos.items()
        if any(a.stem.startswith(prefixo) for a in arquivos_modelos)
    }

    coluna_a, coluna_b = st.columns(2)
    escolha_estacao = coluna_a.selectbox("Estação", estacoes, key="corr_estacao")
    escolha_tipo = coluna_b.selectbox(
        "Tipo de diagnóstico", list(tipos.values()), key="corr_tipo"
    )
    prefixo_tipo = [k for k, v in tipos.items() if v == escolha_tipo][0]

    selecionados = [
        a for a in arquivos_modelos
        if a.stem.startswith(prefixo_tipo) and a.stem.endswith(escolha_estacao)
    ]
    galeria(selecionados, colunas=2)


# ---------------------------------------------------------------- predep

with abas[4]:
    st.subheader("Ranking de preditores")
    st.caption(
        "Mede o quanto conhecer a rajada reduz a incerteza sobre cada covariável, "
        "ao lado das correlações clássicas de Pearson e Spearman."
    )

    ranking = carregar_csv("predep_correlations_results.csv")
    estacoes_ranking = sorted(ranking["Station"].unique())
    escolha = st.selectbox("Estação", estacoes_ranking, key="predep_estacao")

    recorte = ranking[ranking["Station"] == escolha].sort_values("PREDEP", ascending=True)

    grafico = px.bar(
        recorte,
        x="PREDEP",
        y="Feature",
        orientation="h",
        height=max(360, 26 * len(recorte)),
        labels={"PREDEP": "Dependência preditiva", "Feature": "Covariável"},
    )
    grafico.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(grafico, width="stretch", config={"responsive": True})

    st.dataframe(
        recorte.sort_values("PREDEP", ascending=False)[
            ["Feature", "PREDEP", "Pearson", "Spearman", "N_samples"]
        ],
        hide_index=True,
        width="stretch",
    )

    figuras_predep = listar(FIGURAS / "05_predep")
    if figuras_predep:
        st.divider()
        st.markdown("**Ranking consolidado**")
        galeria(figuras_predep, colunas=1)
