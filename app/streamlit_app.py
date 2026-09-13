"""Painel público da Base Vendaval — a trilha de interpolação, do mecanismo ao teto."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
FIGURAS = RAIZ / "figuras"
DADOS = RAIZ / "dados"

st.set_page_config(page_title="Base Vendaval — interpolação de extremos", page_icon="🌬️", layout="wide")


@st.cache_data
def carregar(nome: str) -> pd.DataFrame:
    return pd.read_csv(DADOS / nome)


def fig(nome: str, legenda: str = "") -> None:
    caminho = FIGURAS / nome
    if caminho.exists():
        st.image(str(caminho), caption=legenda or None, width="stretch")


def secao(numero: str, titulo: str, pergunta: str) -> None:
    st.subheader(f"{numero} · {titulo}")
    st.caption(pergunta)


# Ordem de produção, não ordem de qualidade. Fonte: registro de versões do projeto.
VERSOES = [
    ("V1", "Magnitude apenas", "Legado",
     "Resíduo interpolado por IDW somado à magnitude bruta do ERA5. Sem tratar direção do vento."),
    ("V2", "IDW p=2, k=15", "Baseline de produção antigo",
     "IDW clássico (peso 1/d²) nos resíduos, mais interpolação da direção do vento. "
     "Com p=2 o peso cai rápido demais: o campo vira um mosaico quase-constante em torno de cada estação."),
    ("V3", "Gaussiano σ=2,0°, k=15", "Tentativa de corrigir a V2",
     "Troca o peso IDW por kernel gaussiano. Resolveu a transição abrupta, mas perdeu extremos — "
     "σ=2,0° é 13× o default do próprio código."),
    ("V4", "Brown-Resnick calibrado", "Kernel recalibrado",
     "Mesmo mecanismo, com peso calibrado pela escala de dependência medida e distância geodésica "
     "(corrige a distorção de calcular distância direto em graus). Mantém a limitação estrutural."),
    ("V5", "Kriging Ordinário, grid 0,1°", "Candidato de produção",
     "Resolve um sistema linear com todas as estações simultaneamente, em vez de k vizinhos fixos, "
     "num grid ~6× mais fino. Sai da família de combinação convexa de peso fixo."),
]

st.title("🌬️ Base Vendaval")
st.markdown(
    "**Correção da rajada máxima de vento do ERA5 usando as estações do INMET.** "
    "A reanálise cobre o país inteiro em grade regular, mas suaviza o extremo, porque cada célula é "
    "uma média de área. As estações medem o extremo real, em pontos esparsos."
)
st.info(
    "Este painel cobre a **trilha de interpolação**: como espalhar a correção das estações para a grade, "
    "e como as versões dessa etapa foram testadas entre si. A trilha de IA é um trabalho separado.",
    icon="🧭",
)

with st.expander("📁 Onde ficam os dados e o código"):
    st.markdown(
        """
| | Onde |
| --- | --- |
| Dados do projeto | Máquina 3 do cluster de pesquisa, em `/home/publico/vendaval/` |
| Catálogo por arquivo | `docs/MAPA_DADOS.md`, no mesmo diretório |
| Mapa de pastas | `docs/MAPA_DIRETORIOS.md`, no mesmo diretório |
| Código do pipeline | Repositório `Base Vendaval`, no Azure DevOps |

Este painel carrega apenas resultados já produzidos. Nenhum dado bruto é publicado aqui.
        """
    )

st.divider()

abas = st.tabs(
    [
        "1 · O mecanismo",
        "2 · As versões",
        "3 · O teto estrutural",
        "4 · A régua: LOOCV",
        "5 · Para onde foi",
        "Acervo",
    ]
)


# ---------------------------------------------------------------- 1. mecanismo

with abas[0]:
    secao("1", "O mecanismo", "Como se corrige uma grade usando pontos esparsos?")

    esquerda, direita = st.columns([2, 3])
    with esquerda:
        st.markdown(
            """
Três passos, iguais em **todas** as versões:

1. **Resíduo por estação** — `r = rajada_INMET − rajada_ERA5`
2. **Interpolar** o campo de resíduos para a grade inteira
3. **Somar** de volta — `corrigido(x) = ERA5(x) + r̂(x)`

O passo 1 e o 3 nunca mudaram. **O que distingue uma versão da outra é
exclusivamente o passo 2** — e é nele que mora todo este painel.
            """
        )
    with direita:
        fig("01_exploracao/boxplot_gusts_comparison.png")
        st.caption(
            "Por que corrigir: as medianas quase coincidem, mas a estação registra rajadas "
            "muito acima do teto que a reanálise alcança. O erro está na cauda."
        )

    st.divider()
    coluna_a, coluna_b = st.columns(2)
    with coluna_a:
        fig("01_exploracao/inmet_stations_map.png")
        st.caption("A rede de estações: é daqui que sai o resíduo a ser espalhado.")
    with coluna_b:
        fig("01_exploracao/map_station_max_gusts.png")
        st.caption("Rajada máxima observada por estação.")


# ---------------------------------------------------------------- 2. versões

with abas[1]:
    secao("2", "As versões", "O que cada versão mudou no passo de interpolação?")

    for sigla, metodo, status, descricao in VERSOES:
        with st.container(border=True):
            topo, corpo = st.columns([1, 5])
            topo.markdown(f"### {sigla}")
            topo.caption(status)
            corpo.markdown(f"**{metodo}**  \n{descricao}")

    st.caption(
        "A numeração é ordem de produção, não de qualidade — a seção 4 mostra que a ordem "
        "de desempenho é outra."
    )

    st.divider()
    esquerda, direita = st.columns(2)
    with esquerda:
        fig("03_comparacao/v1_vs_v2_maps.png")
        st.caption("V1 e V2 lado a lado.")
    with direita:
        fig("03_comparacao/v3_diffs_p99.png")
        st.caption("O que a V3 mudou em relação à referência, no p99.")


# ---------------------------------------------------------------- 3. teto

with abas[2]:
    secao("3", "O teto estrutural", "Por que V2, V3 e V4 nunca reproduzem um extremo local?")

    esquerda, direita = st.columns([3, 2])
    with esquerda:
        st.markdown(
            """
Os três calculam o resíduo interpolado como **combinação convexa** dos vizinhos:

$$\\hat{r}(x) = \\sum_j w_j(x)\\, r_j, \\qquad w_j \\ge 0, \\quad \\sum_j w_j = 1$$

Toda combinação convexa obedece $\\min_j r_j \\le \\hat{r}(x) \\le \\max_j r_j$.

Ou seja: **o valor interpolado nunca supera o maior resíduo entre os vizinhos usados** — e fica
estritamente abaixo dele sempre que os vizinhos discordarem, que é exatamente o caso quando uma
estação captura um evento extremo e as vizinhas não.

Não é bug de implementação. É propriedade do mecanismo, e vale para qualquer peso — IDW,
gaussiano ou Brown-Resnick.
            """
        )
    with direita:
        st.warning(
            "**A V3 agravou isso por parâmetro.** A escala de dependência espacial real, medida por "
            "F-madograma sobre a série diária das estações, é de **~10 km** (R²≈0,98). A V3 rodou com "
            "σ=2,0°, da ordem de **200 km** — cerca de 20× mais larga. Quanto mais larga a janela, "
            "mais vizinhos discordantes entram na média e mais o extremo se dilui.",
            icon="📏",
        )

    st.divider()
    fig("03_comparacao/dome_200km_p99.png")
    st.caption(
        "Recorte de 200 km ao redor de uma estação (★). Sem suavização a correção fica concentrada; "
        "IDW e kernel gaussiano espalham o efeito. Os dois últimos painéis mostram o que cada um mudou."
    )

    st.markdown("**O trade-off da largura, explícito:**")
    fig("03_comparacao/tune_sigma_grid.png")
    st.caption(
        "Varredura da largura do kernel. À esquerda, detalhe local e transições abruptas. "
        "À direita, o campo achata e a informação da estação se dissolve."
    )


# ---------------------------------------------------------------- 4. LOOCV

with abas[3]:
    secao("4", "A régua: LOOCV", "Qual método realmente prevê melhor onde não há estação?")

    st.markdown(
        "**Validação cruzada leave-one-station-out:** cada estação é removida, seu valor é predito "
        "usando só as demais, e comparado ao observado. É a única régua que mede o que importa — "
        "acertar onde *não* há medição."
    )

    metricas = carregar("loocv_consolidated_metrics.csv")
    percentil = st.radio("Percentil", ["p99", "p95"], horizontal=True, key="loocv_pct")
    recorte = metricas[metricas["pct"] == percentil].sort_values("rmse").reset_index(drop=True)

    esquerda, direita = st.columns([3, 2])
    with esquerda:
        grafico = px.bar(
            recorte.sort_values("rmse", ascending=False),
            x="rmse",
            y="method",
            orientation="h",
            color="bias",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            height=420,
            labels={"rmse": "RMSE (m/s) — menor é melhor", "method": "", "bias": "Viés"},
        )
        grafico.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(grafico, width="stretch", config={"responsive": True})
    with direita:
        exibir = recorte[["method", "bias", "rmse", "mae", "corr"]].copy()
        exibir.columns = ["Método", "Viés", "RMSE", "MAE", "Corr"]
        for coluna in ["Viés", "RMSE", "MAE", "Corr"]:
            exibir[coluna] = exibir[coluna].round(2)
        st.dataframe(exibir, hide_index=True, width="stretch")
        st.caption(f"Todos em m/s, exceto correlação. N = {int(recorte['n'].iloc[0])} estações.")

    producao = recorte[recorte["method"].str.startswith(("V2", "V3", "V4"))]
    if not producao.empty:
        piores = recorte.tail(len(recorte) // 2)["method"].tolist()
        st.error(
            f"**As versões de produção são as piores da lista.** V2, V3 e V4 aparecem com viés de "
            f"{producao['bias'].min():.1f} a {producao['bias'].max():.1f} m/s — subestimando o extremo "
            "por larga margem, exatamente o que a seção 3 prevê.",
            icon="⚠️",
        )
    melhor = recorte.iloc[0]
    st.success(
        f"**{melhor['method']}** lidera entre os métodos rápidos (RMSE {melhor['rmse']:.2f} m/s, "
        f"viés {melhor['bias']:+.2f}). Vale notar: é uma variante do próprio IDW, com o expoente "
        "calibrado localmente em vez de fixo — parte do problema da V2 era o parâmetro, não o mecanismo.",
        icon="✅",
    )

    with st.expander("Simulação condicional (MSP2) — melhor resultado, em subamostra de 62 estações"):
        msp2 = carregar("msp2_loocv_fmadograma_p99.csv")
        erro = msp2["predicted"] - msp2["observed"]
        coluna_a, coluna_b, coluna_c = st.columns(3)
        coluna_a.metric("Viés", f"{erro.mean():+.2f} m/s")
        coluna_b.metric("RMSE", f"{(erro**2).mean()**0.5:.2f} m/s")
        coluna_c.metric("Correlação", f"{msp2['predicted'].corr(msp2['observed']):.2f}")
        dispersao = px.scatter(
            msp2, x="observed", y="predicted", hover_name="codigo_estacao", height=380,
            labels={"observed": "Observado (m/s)", "predicted": "Predito (m/s)"},
        )
        limite = [float(min(msp2["observed"].min(), msp2["predicted"].min())),
                  float(max(msp2["observed"].max(), msp2["predicted"].max()))]
        dispersao.add_shape(type="line", x0=limite[0], y0=limite[0], x1=limite[1], y1=limite[1],
                            line=dict(dash="dash", color="#888", width=1))
        dispersao.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(dispersao, width="stretch", config={"responsive": True})
        st.caption(
            "Simulação Monte Carlo do processo max-estável completo, condicionando em várias estações "
            "ao mesmo tempo. É o melhor resultado medido, mas custa minutos por ponto — serve como "
            "ferramenta de auditoria pontual, não para gerar a grade histórica inteira."
        )


# ---------------------------------------------------------------- 5. destino

with abas[4]:
    secao("5", "Para onde foi", "O que essa trilha concluiu?")

    st.markdown(
        """
Duas conclusões, uma metodológica e uma de rumo.

**A interpolação por combinação convexa tem teto.** Recalibrar o kernel (V4) ou trocar o peso (V3)
não resolve, porque o limite é do mecanismo, não do parâmetro. Os métodos que se saíram melhor são
justamente os que saem dessa família: mirar o quantil diretamente, adaptar o expoente localmente,
ou simular o processo extremal.

**O rumo do projeto passou a ser a trilha de IA.** A base de IA desenvolvida em paralelo foi
comparada contra a interpolação em pé de igualdade — mesmo recorte espacial, mesmo período fora da
amostra de treino, e com validação cruzada real do lado da interpolação — e levou vantagem. É por
isso que o trabalho seguiu por lá.
        """
    )
    st.caption(
        "A comparação em pé de igualdade é deliberada: a versão de produção usa a rede nacional inteira "
        "e o histórico completo, vantagem de informação que a base de IA não tinha. Por isso a régua "
        "usou uma versão restrita ao mesmo domínio e período."
    )


# ---------------------------------------------------------------- acervo

with abas[5]:
    st.subheader("Acervo")
    st.caption("Figuras produzidas pelo pipeline, inclusive as que não entraram no percurso acima.")

    grupos = {
        "Comparação entre versões e ajuste": sorted((FIGURAS / "03_comparacao").glob("*.png")),
        "Exploração — campos médios da reanálise": sorted((FIGURAS / "01_exploracao").glob("*_mean_map.png")),
        "Exploração — covariáveis por estação": sorted((FIGURAS / "01_exploracao").glob("map_covariate_*.png")),
        "Exploração — recorte de interesse": sorted((FIGURAS / "01_exploracao").glob("map_roi_*.png")),
        "Exploração — distribuições": sorted((FIGURAS / "01_exploracao").glob("boxplot_*.png"))
        + sorted((FIGURAS / "01_exploracao").glob("pdf_*.png"))
        + sorted((FIGURAS / "01_exploracao").glob("heatmap_*.png")),
    }
    grupos = {t: a for t, a in grupos.items() if a}

    # Um grupo por vez: o Streamlit desenha conteúdo de expander mesmo fechado.
    escolhido = st.selectbox(
        "Grupo", list(grupos), format_func=lambda t: f"{t} ({len(grupos[t])})", key="acervo_grupo"
    )
    arquivos = grupos[escolhido]
    for inicio in range(0, len(arquivos), 3):
        for coluna, arquivo in zip(st.columns(3), arquivos[inicio : inicio + 3]):
            coluna.image(str(arquivo), caption=arquivo.stem.replace("_", " "), width="stretch")
