"""Painel público do projeto Vendaval — o percurso da análise, do problema ao resultado."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
FIGURAS = RAIZ / "figuras"
DADOS = RAIZ / "dados"

st.set_page_config(page_title="Vendaval — correção de rajada de vento", page_icon="🌬️", layout="wide")


# ---------------------------------------------------------------- utilidades


@st.cache_data
def carregar_csv(nome: str) -> pd.DataFrame:
    return pd.read_csv(DADOS / nome)


@st.cache_data
def metricas_por_versao(percentil: str) -> pd.DataFrame:
    """Erro de cada versão do campo contra a estação, calculado na hora."""
    dados = carregar_csv("scatter_p95_p99_dados.csv")
    alvo = f"inmet_{percentil}"
    linhas = []
    for nome, coluna in VERSOES.items():
        recorte = dados[[alvo, f"{coluna}_{percentil}"]].dropna()
        erro = recorte[f"{coluna}_{percentil}"] - recorte[alvo]
        linhas.append(
            {
                "Versão": nome,
                "Viés": round(erro.mean(), 2),
                "EAM": round(erro.abs().mean(), 2),
                "REQM": round((erro**2).mean() ** 0.5, 2),
                "n": len(recorte),
            }
        )
    return pd.DataFrame(linhas)


VERSOES = {"ERA5 original": "era_orig", "V2 — IDW": "v2", "V3 — Gaussiano": "v3"}


def fig(nome: str, legenda: str = "", largura: str = "stretch") -> None:
    caminho = FIGURAS / nome
    if caminho.exists():
        st.image(str(caminho), caption=legenda or None, width=largura)


def secao(numero: str, titulo: str, pergunta: str) -> None:
    st.subheader(f"{numero} · {titulo}")
    st.caption(pergunta)


def achado(texto: str) -> None:
    st.success(texto, icon="✅")


# ---------------------------------------------------------------- cabeçalho

st.title("🌬️ Vendaval")
st.markdown(
    "**Correção da rajada máxima de vento da reanálise ERA5 usando as estações do INMET.** "
    "A reanálise cobre o país inteiro em grade regular, mas suaviza o extremo. "
    "As estações medem o extremo de verdade, só que em pontos esparsos. "
    "O projeto usa as estações para corrigir a grade."
)

with st.expander("📁 Onde ficam os dados e o código"):
    st.markdown(
        """
| | Onde |
| --- | --- |
| Dados do projeto | Máquina 3 do cluster de pesquisa, em `/home/publico/vendaval/` |
| Catálogo por arquivo | `docs/MAPA_DADOS.md`, no mesmo diretório — o que é cada dado, qual script gerou e de qual dado nasceu |
| Mapa de pastas | `docs/MAPA_DIRETORIOS.md`, no mesmo diretório |
| Código do pipeline | Repositório `Base Vendaval`, no Azure DevOps |

Este painel carrega apenas resultados já produzidos. Nenhum dado bruto é publicado aqui.
        """
    )

st.divider()

abas = st.tabs(
    [
        "1 · O problema",
        "2 · Os dados",
        "3 · O que explica a rajada",
        "4 · Os modelos",
        "5 · Do ponto à grade",
        "6 · O resultado",
        "Acervo",
    ]
)


# ---------------------------------------------------------------- 1. problema

with abas[0]:
    secao("1", "O problema", "A reanálise erra a rajada? Onde exatamente?")

    esquerda, direita = st.columns([3, 2])
    with esquerda:
        fig("01_exploracao/boxplot_gusts_comparison.png")
    with direita:
        st.markdown(
            "As duas distribuições têm **mediana quase igual**. A diferença está na cauda: "
            "a estação registra rajadas bem acima do teto que a reanálise alcança.\n\n"
            "Ou seja, o ERA5 acerta o vento típico e erra justamente o evento raro — "
            "que é o que interessa em análise de vendaval."
        )
        tabela = metricas_por_versao("p99")
        vies_era5 = tabela.loc[tabela["Versão"] == "ERA5 original", "Viés"].iloc[0]
        vies_p95 = metricas_por_versao("p95").loc[lambda t: t["Versão"] == "ERA5 original", "Viés"].iloc[0]
        achado(
            f"Medido sobre {int(tabela['n'].iloc[0])} estações: o ERA5 subestima o p95 em "
            f"{abs(vies_p95):.2f} m/s e o p99 em {abs(vies_era5):.2f} m/s. "
            "Quanto mais extremo o quantil, maior o erro."
        )

    st.caption("É esse buraco na cauda que o resto do projeto tenta fechar.")


# ---------------------------------------------------------------- 2. dados

with abas[1]:
    secao("2", "Os dados", "O que entra no pipeline, e com que cobertura?")

    esquerda, direita = st.columns(2)
    with esquerda:
        fig("01_exploracao/inmet_stations_map.png")
        st.caption("Estações automáticas do INMET sobre a malha do ERA5 — a referência observada.")
    with direita:
        fig("01_exploracao/map_station_max_gusts.png")
        st.caption("Rajada máxima registrada por estação. Define o alvo que o modelo precisa reproduzir.")

    st.markdown(
        "**A assimetria que define o método:** a reanálise é contínua e cobre tudo; "
        "a observação é pontual e esparsa. Corrigir significa aprender o erro onde há estação "
        "e propagar esse aprendizado para onde não há — o que vira o problema da seção 5."
    )


# ---------------------------------------------------------------- 3. covariáveis

with abas[2]:
    secao("3", "O que explica a rajada", "Quais variáveis carregam informação sobre o erro?")

    st.markdown(
        "Antes de treinar qualquer modelo, é preciso saber quais covariáveis meteorológicas "
        "têm relação real com a rajada observada. Duas leituras independentes:"
    )

    esquerda, direita = st.columns(2)
    with esquerda:
        st.markdown("**Correlação clássica**")
        fig("01_exploracao/heatmap_correlation.png")
        st.caption("Correlação entre as covariáveis candidatas — revela redundância entre elas.")
    with direita:
        st.markdown("**Dependência preditiva (PREDEP)**")
        st.caption(
            "Mede o quanto conhecer a rajada reduz a incerteza sobre a covariável. "
            "Captura relação não linear, que a correlação linear perde."
        )
        ranking = carregar_csv("predep_correlations_results.csv")
        estacao = st.selectbox("Estação", sorted(ranking["Station"].unique()), key="predep_estacao")
        recorte = ranking[ranking["Station"] == estacao].sort_values("PREDEP")
        grafico = px.bar(
            recorte,
            x="PREDEP",
            y="Feature",
            orientation="h",
            height=max(340, 24 * len(recorte)),
            labels={"PREDEP": "Dependência preditiva", "Feature": ""},
        )
        grafico.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(grafico, width="stretch", config={"responsive": True})

    primeiras = (
        ranking.sort_values("PREDEP", ascending=False).groupby("Station").head(1)["Feature"].tolist()
    )
    achado(
        "Em todas as estações avaliadas a primeira colocada é vento ou precipitação "
        f"({', '.join(sorted(set(primeiras)))}) — nenhuma variável termodinâmica chega ao topo."
    )

    with st.expander("Tabela completa do ranking"):
        st.dataframe(
            ranking[ranking["Station"] == estacao]
            .sort_values("PREDEP", ascending=False)[["Feature", "PREDEP", "Pearson", "Spearman", "N_samples"]],
            hide_index=True,
            width="stretch",
        )


# ---------------------------------------------------------------- 4. modelos

with abas[3]:
    secao("4", "Os modelos", "Qual família de modelo reproduz melhor a rajada observada?")

    st.markdown(
        "Três famílias treinadas **por estação**, não um modelo único nacional: regressão linear "
        "regularizada, floresta aleatória e regressão quantílica. A quantílica entra porque o alvo "
        "é um extremo — ajustar a média não garante ajustar a cauda."
    )

    familias = {
        "Linear e floresta aleatória": FIGURAS / "04_correcao/regressao",
        "Regressão quantílica": FIGURAS / "04_correcao/quantilica",
    }
    diagnosticos = {
        "scatter": "Previsto × observado",
        "residuals": "Resíduos",
        "importance": "Importância das variáveis",
        "multi": "Múltiplos quantis",
    }

    coluna_a, coluna_b, coluna_c = st.columns(3)
    familia = coluna_a.selectbox("Família", list(familias), key="mod_familia")
    pasta = familias[familia]
    arquivos = sorted(pasta.glob("*.png"))

    disponiveis = {p: r for p, r in diagnosticos.items() if any(a.stem.startswith(p) for a in arquivos)}
    estacao_modelo = coluna_b.selectbox(
        "Estação", sorted({a.stem.split("_")[-1] for a in arquivos}), key="mod_estacao"
    )
    rotulo = coluna_c.selectbox("Diagnóstico", list(disponiveis.values()), key="mod_diag")
    prefixo = [p for p, r in disponiveis.items() if r == rotulo][0]

    selecionados = [a for a in arquivos if a.stem.startswith(prefixo) and a.stem.endswith(estacao_modelo)]
    colunas = st.columns(min(2, len(selecionados)) or 1)
    for coluna, arquivo in zip(colunas, selecionados):
        coluna.image(str(arquivo), caption=arquivo.stem.replace("_", " "), width="stretch")

    st.caption(
        "Ler o *previsto × observado*: pontos abaixo da diagonal são eventos que o modelo "
        "subestimou. A dispersão cresce nos valores altos — o extremo continua sendo a parte difícil."
    )


# ---------------------------------------------------------------- 5. ponto → grade

with abas[4]:
    secao("5", "Do ponto à grade", "A correção nasce nas estações. Como virar campo contínuo sem inventar estrutura?")

    st.markdown(
        "Aplicar a correção só onde há estação cria **ilhas**: cada estação vira um ponto quente "
        "cercado de grade não corrigida. A saída é interpolar — e a pergunta passa a ser **quanto** suavizar."
    )

    fig("03_comparacao/dome_200km_p99.png")
    st.caption(
        "Recorte de 200 km ao redor de uma estação (★). Sem suavização, a correção fica concentrada; "
        "IDW e kernel gaussiano espalham o efeito. Os dois últimos painéis mostram o que cada método mudou."
    )

    st.divider()
    st.markdown("**O trade-off, explícito:**")
    fig("03_comparacao/tune_sigma_grid.png")
    st.caption(
        "Varredura da largura do kernel gaussiano. À esquerda, detalhe local preservado e transições abruptas. "
        "À direita, o campo é achatado e a informação da estação se dissolve num alcance de centenas de quilômetros."
    )

    achado(
        "Suavizar demais não é um defeito estético — apaga a correção. É o que a seção 6 mede."
    )

    with st.expander("Efeito em escala nacional"):
        fig("03_comparacao/compare_quantis_3x2.png")
        st.caption("V2 base e as duas suavizações, em p95 e p99, sobre o Brasil inteiro.")


# ---------------------------------------------------------------- 6. resultado

with abas[5]:
    secao("6", "O resultado", "A correção funcionou? De quanto foi o ganho?")

    percentil = st.radio("Percentil", ["p95", "p99"], horizontal=True, key="res_percentil")
    dados = carregar_csv("scatter_p95_p99_dados.csv")
    alvo = f"inmet_{percentil}"

    figura = go.Figure()
    for nome, coluna in VERSOES.items():
        recorte = dados[[alvo, f"{coluna}_{percentil}", "codigo_estacao"]].dropna()
        figura.add_trace(
            go.Scattergl(
                x=recorte[alvo],
                y=recorte[f"{coluna}_{percentil}"],
                mode="markers",
                name=nome,
                marker=dict(size=5, opacity=0.55),
                customdata=recorte["codigo_estacao"],
                hovertemplate="%{customdata}<br>observado %{x:.1f}<br>campo %{y:.1f} m/s<extra></extra>",
            )
        )
    limite = [
        float(min(dados[alvo].min(), dados[[f"{c}_{percentil}" for c in VERSOES.values()]].min().min())),
        float(max(dados[alvo].max(), dados[[f"{c}_{percentil}" for c in VERSOES.values()]].max().max())),
    ]
    figura.add_trace(
        go.Scatter(x=limite, y=limite, mode="lines", name="acerto perfeito",
                   line=dict(dash="dash", width=1, color="#888"), hoverinfo="skip")
    )
    figura.update_layout(
        height=520,
        xaxis_title=f"Observado na estação — {percentil} (m/s)",
        yaxis_title=f"Campo — {percentil} (m/s)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    figura.update_xaxes(range=limite)
    figura.update_yaxes(range=limite)

    esquerda, direita = st.columns([3, 2])
    with esquerda:
        st.plotly_chart(figura, width="stretch", config={"responsive": True})
    with direita:
        tabela = metricas_por_versao(percentil)
        st.dataframe(tabela, hide_index=True, width="stretch")
        st.caption(
            "Todos em m/s. EAM = erro absoluto médio, REQM = raiz do erro quadrático médio, "
            "n = estações. Viés negativo = campo abaixo da estação."
        )

        v_era5 = tabela.loc[tabela["Versão"] == "ERA5 original", "Viés"].iloc[0]
        v_v2 = tabela.loc[tabela["Versão"] == "V2 — IDW", "Viés"].iloc[0]
        v_v3 = tabela.loc[tabela["Versão"] == "V3 — Gaussiano", "Viés"].iloc[0]
        reducao = (1 - abs(v_v2) / abs(v_era5)) * 100
        achado(
            f"**V2 reduz o viés em {reducao:.0f}%** ({v_era5:+.2f} → {v_v2:+.2f} m/s), "
            "com queda equivalente no erro absoluto."
        )
        st.error(
            f"**V3 fica pior que o ponto de partida** ({v_v3:+.2f} m/s contra {v_era5:+.2f}). "
            "O kernel largo da seção 5 dissolveu a correção — o campo volta a subestimar.",
            icon="⚠️",
        )

    st.markdown(
        "**Leitura:** a correção por estação funciona; o que decide o resultado final é a etapa de "
        "espacialização. V2 (IDW) preserva o ganho, V3 (gaussiano largo) o devolve."
    )
    st.caption(
        "Ressalva: são percentis históricos por estação, não erro evento a evento. "
        "As estações usadas no ajuste também aparecem aqui."
    )


# ---------------------------------------------------------------- acervo

with abas[6]:
    st.subheader("Acervo completo")
    st.caption("Todas as figuras produzidas pelo pipeline, inclusive as que não entraram no percurso acima.")

    grupos = {
        "Exploração — campos médios da reanálise": sorted(
            p for p in (FIGURAS / "01_exploracao").glob("*_mean_map.png")
        ),
        "Exploração — covariáveis por estação": sorted((FIGURAS / "01_exploracao").glob("map_covariate_*.png")),
        "Exploração — recorte de interesse": sorted((FIGURAS / "01_exploracao").glob("map_roi_*.png")),
        "Exploração — distribuições": sorted((FIGURAS / "01_exploracao").glob("boxplot_*.png"))
        + sorted((FIGURAS / "01_exploracao").glob("pdf_*.png")),
        "Baseline": sorted((FIGURAS / "02_baseline").glob("*.png")),
        "Comparação entre versões e ajuste": sorted((FIGURAS / "03_comparacao").glob("*.png")),
        "Diagnóstico dos modelos": sorted((FIGURAS / "04_correcao").rglob("*.png")),
        "Ranking de preditores": sorted((FIGURAS / "05_predep").glob("*.png")),
    }

    # Um grupo por vez: o Streamlit desenha o conteúdo de expander mesmo fechado, e
    # montar as 101 figuras de uma vez deixa a página pesada sem necessidade.
    grupos = {t: a for t, a in grupos.items() if a}
    escolhido = st.selectbox(
        "Grupo", list(grupos), format_func=lambda t: f"{t} ({len(grupos[t])})", key="acervo_grupo"
    )
    arquivos = grupos[escolhido]
    for inicio in range(0, len(arquivos), 3):
        for coluna, arquivo in zip(st.columns(3), arquivos[inicio : inicio + 3]):
            coluna.image(str(arquivo), caption=arquivo.stem.replace("_", " "), width="stretch")
