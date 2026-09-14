"""Painel público da Base Vendaval — a trilha de interpolação, do mecanismo ao teto."""

import struct
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
FIGURAS = RAIZ / "figuras"
DADOS = RAIZ / "dados"

st.set_page_config(
    page_title="Base Vendaval — interpolação de extremos",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  .block-container { padding-top: 2.2rem; max-width: 1500px; }
  h1, h2, h3 { letter-spacing: -0.01em; }
  .hero {
    border: 1px solid rgba(57,135,229,.35);
    border-left: 4px solid #3987e5;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    background: linear-gradient(100deg, rgba(57,135,229,.11), rgba(57,135,229,.02));
    margin-bottom: .4rem;
  }
  .hero h1 { margin: 0 0 .45rem 0; font-size: 2.1rem; }
  .hero p { margin: 0; opacity: .88; font-size: 1.02rem; line-height: 1.55; }
  .passo {
    border-left: 3px solid #3987e5; padding: .15rem 0 .15rem .8rem;
    margin-bottom: .7rem;
  }
  .passo b { color: #6aa8f0; }
  div[data-testid="stMetricValue"] { font-size: 1.5rem; }
</style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- dados


@st.cache_data
def carregar(nome: str) -> pd.DataFrame:
    return pd.read_csv(DADOS / nome)


MAPAS_025 = {
    "V2 (IDW p=2, k=15)": "v2",
    "V3 (Gaussiano σ=2,0°, k=15)": "v3",
    "V4 (Brown-Resnick calibrado)": "v4",
    "IDW Adaptativo": "idw_adapt",
    "Quantile Kriging": "qk",
    "RBF Multiquadric": "rbf",
    "EVK": "evk",
    "MSP1 (F-madograma)": "msp1_fmadograma",
    "MSP1 (Verossimilhança composta)": "msp1_verossimilhanca",
}


@st.cache_resource
def pacote_de_mapas():
    return dict(np.load(DADOS / "mapas_espaciais.npz", allow_pickle=False))


def campo_mascarado(chave: str, sufixo: str) -> np.ndarray | None:
    pacote = pacote_de_mapas()
    if chave not in pacote:
        return None
    campo = pacote[chave].astype("float64").copy()
    mascara = pacote[f"mask_{sufixo}"]
    campo[~mascara] = np.nan
    return campo


def mapa(campo: np.ndarray, sufixo: str, titulo: str, divergente: bool = False):
    """Heatmap geográfico com 1° de longitude = 1° de latitude.

    A trava de proporção é segura aqui porque a navegação é por estado: seção
    não selecionada não é montada, então nunca há container escondido sem
    largura (a armadilha de primeiro desenho do Streamlit).
    """
    pacote = pacote_de_mapas()
    lats, lons = pacote[f"lats_{sufixo}"], pacote[f"lons_{sufixo}"]
    validos = campo[~np.isnan(campo)]
    if divergente:
        limite = float(np.percentile(np.abs(validos), 95)) if validos.size else 1.0
        zmin, zmax, escala = -limite, limite, "RdBu_r"
    else:
        zmin = float(validos.min()) if validos.size else 0.0
        zmax = float(validos.max()) if validos.size else 1.0
        escala = "Turbo"
    figura_mapa = go.Figure(
        go.Heatmap(z=campo, x=lons, y=lats, zmin=zmin, zmax=zmax, colorscale=escala,
                   hovertemplate="lon %{x:.2f}<br>lat %{y:.2f}<br>%{z:.2f} m/s<extra></extra>",
                   colorbar=dict(title="m/s"))
    )
    figura_mapa.add_trace(
        go.Scatter(x=pacote["contorno_x"], y=pacote["contorno_y"], mode="lines",
                   line=dict(color="rgba(255,255,255,.45)", width=1),
                   hoverinfo="skip", showlegend=False)
    )
    figura_mapa.update_layout(
        title=titulo, height=560, margin=dict(l=10, r=10, t=48, b=10),
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(title=None, showgrid=False, scaleanchor="x", scaleratio=1),
    )
    return figura_mapa


@st.cache_data
def v5_por_estacao(percentil: str) -> pd.DataFrame:
    """V5 nas estações, nas duas leituras, alinhada ao observado do LOOCV.

    `verdadeira` exclui a estação-alvo do próprio sistema de kriging;
    `producao` é a grade materializada, amostrada na estação usando ela mesma.
    A diferença entre as duas é o tamanho do vazamento.
    """
    predicoes = carregar("loocv_per_station_predictions.csv")
    observado = (
        predicoes[predicoes["pct"] == percentil][["codigo_estacao", "observed"]].drop_duplicates()
    )
    saida = []
    for rotulo, arquivo in [
        ("V5 (LOOCV real)", "v5_verdadeira_loocv.csv"),
        ("V5 (amostrada na produção)", "v5_producao_amostrada.csv"),
    ]:
        fonte = carregar(arquivo)[["codigo_estacao", percentil]].dropna()
        juntos = fonte.merge(observado, on="codigo_estacao")
        juntos = juntos.rename(columns={percentil: "predicted"})
        juntos["method"] = rotulo
        saida.append(juntos[["method", "codigo_estacao", "predicted", "observed"]])
    return pd.concat(saida, ignore_index=True)


def erro_resumido(df: pd.DataFrame) -> dict:
    erro = df["predicted"] - df["observed"]
    return {
        "bias": erro.mean(),
        "rmse": (erro**2).mean() ** 0.5,
        "mae": erro.abs().mean(),
        "corr": df["predicted"].corr(df["observed"]),
        "n": len(df),
    }


@st.cache_data
def proporcao(caminho: str) -> float:
    """Largura/altura lida do cabeçalho do PNG — sem decodificar a imagem."""
    with open(caminho, "rb") as arquivo:
        cabecalho = arquivo.read(33)
    largura, altura = struct.unpack(">II", cabecalho[16:24])
    return largura / altura


def figura(nome: str, legenda: str = "") -> None:
    """Desenha a figura com a largura que o formato dela pede.

    Multipainel largo (tira horizontal) precisa de largura inteira; retrato alto
    fica gigante em largura inteira e precisa ser contido. Meia largura só serve
    para figura de painel único, com proporção próxima de quadrada.
    """
    caminho = FIGURAS / nome
    if not caminho.exists():
        st.info(f"Figura ausente: `{nome}`.")
        return
    aspecto = proporcao(str(caminho))
    if aspecto < 1.0:  # retrato — conter, senão domina a tela inteira
        _, meio, _ = st.columns([1, 2, 1])
        alvo = meio
    else:
        alvo = st.container()
    with alvo:
        st.image(str(caminho), caption=legenda or None, width="stretch")


def grade_de_figuras(arquivos: list[Path], legenda_por_nome: bool = True) -> None:
    """Agrupa figuras em linhas, com quantidade por linha vinda da proporção."""
    fila: list[Path] = []

    def descarregar(itens: list[Path], por_linha: int) -> None:
        for inicio in range(0, len(itens), por_linha):
            lote = itens[inicio : inicio + por_linha]
            colunas = st.columns(por_linha)
            for coluna, arquivo in zip(colunas, lote):
                with coluna:
                    st.image(
                        str(arquivo),
                        caption=arquivo.stem.replace("_", " ") if legenda_por_nome else None,
                        width="stretch",
                    )

    for arquivo in arquivos:
        aspecto = proporcao(str(arquivo))
        if aspecto >= 2.2 or aspecto < 1.0:
            descarregar(fila, 2)
            fila = []
            st.image(
                str(arquivo),
                caption=arquivo.stem.replace("_", " ") if legenda_por_nome else None,
                width="stretch",
            )
        else:
            fila.append(arquivo)
    descarregar(fila, 2)


SECOES = [
    ("inicio", "🏠 Início", "", ""),
    ("mecanismo", "1 · O mecanismo", "Como se corrige uma grade usando pontos esparsos?",
     "Os três passos que nunca mudaram, e o único que muda entre versões."),
    ("versoes", "2 · As versões", "O que cada versão mudou no passo de interpolação?",
     "De V1 a V5, com o que cada uma tentou resolver e o que produziu."),
    ("teto", "3 · O teto estrutural", "Por que V2, V3 e V4 nunca reproduzem um extremo local?",
     "A conta que limita o mecanismo, e a escala espacial medida nos dados."),
    ("loocv", "4 · A régua: LOOCV", "Qual método realmente prevê melhor onde não há estação?",
     "Nove métodos na mesma validação cruzada, com controle de vazamento."),
    ("mapas", "5 · Os mapas", "O que cada método produz no território?",
     "O campo corrigido de cada método e o que ele muda em relação à produção."),
    ("destino", "6 · Para onde foi", "O que essa trilha concluiu?",
     "A conclusão metodológica e a comparação ainda em aberto."),
    ("acervo", "📚 Acervo", "", "Todas as figuras produzidas pelo pipeline."),
]
ROTULOS = {chave: rotulo for chave, rotulo, _, _ in SECOES}

if "secao" not in st.session_state:
    st.session_state.secao = "inicio"


def ir_para(destino: str) -> None:
    """Navegação do hub.

    Precisa rodar como callback de botão (`on_click`), nunca no corpo do script:
    `secao` é a chave do rádio da barra lateral, e o Streamlit proíbe escrever
    numa chave de widget já instanciado no mesmo ciclo. Callback executa antes
    do próximo ciclo, quando nenhum widget existe ainda.
    """
    st.session_state.secao = destino


with st.sidebar:
    st.markdown("### 🌬️ Base Vendaval")
    st.caption("Correção de rajada por interpolação")
    st.radio(
        "Seções",
        [chave for chave, _, _, _ in SECOES],
        format_func=lambda chave: ROTULOS[chave],
        key="secao",
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(
        "Passe o mouse sobre qualquer figura e use o ícone de expandir "
        "para vê-la em tela cheia."
    )

secao_atual = st.session_state.secao


def cabecalho_da_secao(chave: str) -> None:
    rotulo, pergunta = ROTULOS[chave], dict((c, p) for c, _, p, _ in SECOES)[chave]
    st.subheader(rotulo)
    if pergunta:
        st.caption(pergunta)


# ---------------------------------------------------------------- início

if secao_atual == "inicio":
    st.markdown(
        """
<div class="hero">
  <h1>🌬️ Base Vendaval</h1>
  <p><b>Correção da rajada máxima de vento do ERA5 usando as estações do INMET.</b><br>
  A reanálise cobre o país inteiro em grade regular, mas suaviza o extremo, porque cada célula
  é uma média de área. As estações medem o extremo real, em pontos esparsos. O projeto usa umas
  para corrigir a outra.</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Este painel cobre a **trilha de interpolação**: como espalhar a correção das estações "
        "para a grade, e como as versões dessa etapa foram testadas entre si. "
        "A trilha de IA é um trabalho separado.",
        icon="🧭",
    )

    metricas = carregar("loocv_consolidated_metrics.csv")
    p99 = metricas[metricas["pct"] == "p99"]
    colunas = st.columns(4)
    colunas[0].metric("Versões da correção", "5", help="V1 a V5, ordem de produção")
    colunas[1].metric("Métodos comparados", str(p99["method"].nunique()))
    colunas[2].metric("Estações na validação", f"{int(p99['n'].max())}")
    colunas[3].metric("Escala espacial medida", "~10 km", help="F-madograma, R²≈0,98")

    st.divider()
    st.markdown("#### Por onde começar")

    navegaveis = [s for s in SECOES if s[0] not in ("inicio", "acervo")]
    for inicio in range(0, len(navegaveis), 2):
        linha = st.columns(2)
        for coluna, (chave, rotulo, pergunta, resumo) in zip(linha, navegaveis[inicio : inicio + 2]):
            with coluna, st.container(border=True):
                st.markdown(f"**{rotulo}**")
                st.caption(pergunta)
                st.markdown(resumo)
                st.button(
                    "Abrir", key=f"ir_{chave}", width="stretch",
                    on_click=ir_para, args=(chave,),
                )

    with st.container(border=True):
        st.markdown("**📚 Acervo**")
        st.markdown("Todas as figuras produzidas pelo pipeline, agrupadas por tema.")
        st.button(
            "Abrir", key="ir_acervo", width="stretch",
            on_click=ir_para, args=("acervo",),
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


# ---------------------------------------------------------------- mecanismo

elif secao_atual == "mecanismo":
    cabecalho_da_secao("mecanismo")

    esquerda, direita = st.columns([2, 3])
    with esquerda:
        st.markdown(
            """
<div class="passo"><b>1. Resíduo por estação</b><br>
<code>r = rajada_INMET − rajada_ERA5</code></div>
<div class="passo"><b>2. Interpolar</b><br>
o campo de resíduos para a grade inteira</div>
<div class="passo"><b>3. Somar de volta</b><br>
<code>corrigido(x) = ERA5(x) + r̂(x)</code></div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "Os passos 1 e 3 nunca mudaram. **O que distingue uma versão da outra é "
            "exclusivamente o passo 2** — e é nele que mora todo este painel."
        )
    with direita:
        figura(
            "01_exploracao/boxplot_gusts_comparison.png",
            "Por que corrigir: as medianas quase coincidem, mas a estação registra rajadas muito "
            "acima do teto que a reanálise alcança. O erro está na cauda.",
        )

    st.divider()
    st.markdown("**De onde vem o resíduo**")
    grade_de_figuras(
        [FIGURAS / "01_exploracao/inmet_stations_map.png",
         FIGURAS / "01_exploracao/map_station_max_gusts.png"],
        legenda_por_nome=False,
    )
    st.caption("À esquerda, a rede de estações. À direita, a rajada máxima observada em cada uma.")


# ---------------------------------------------------------------- versões

elif secao_atual == "versoes":
    cabecalho_da_secao("versoes")

    versoes = [
        ("V1", "Magnitude apenas", "Legado",
         "Resíduo interpolado por IDW somado à magnitude bruta do ERA5. Sem tratar direção do vento."),
        ("V2", "IDW p=2, k=15", "Baseline de produção antigo",
         "IDW clássico (peso 1/d²) nos resíduos, mais interpolação da direção. Com p=2 o peso cai "
         "rápido demais: o campo vira um mosaico quase-constante em torno de cada estação."),
        ("V3", "Gaussiano σ=2,0°, k=15", "Tentativa de corrigir a V2",
         "Troca o peso IDW por kernel gaussiano. Resolveu a transição abrupta, mas perdeu extremos — "
         "σ=2,0° é 13× o default do próprio código."),
        ("V4", "Brown-Resnick calibrado", "Kernel recalibrado",
         "Peso calibrado pela escala de dependência medida, com distância geodésica — corrige a "
         "distorção de calcular distância direto em graus. Mantém a limitação estrutural."),
        ("V5", "Kriging Ordinário, grid 0,1°", "Candidato de produção",
         "Resolve um sistema linear com todas as estações simultaneamente, em vez de k vizinhos "
         "fixos, num grid ~6× mais fino. Sai da família de peso fixo."),
    ]
    for sigla, metodo, status, descricao in versoes:
        with st.container(border=True):
            topo, corpo = st.columns([1, 6])
            topo.markdown(f"### {sigla}")
            topo.caption(status)
            corpo.markdown(f"**{metodo}**  \n{descricao}")

    st.caption("A numeração é ordem de produção, não de qualidade — a seção 4 mostra a ordem real.")

    st.divider()
    st.markdown("#### O que cada versão produz, estação a estação")

    predicoes = carregar("loocv_per_station_predictions.csv")
    coluna_a, coluna_b = st.columns([2, 1])
    percentil = coluna_b.radio("Percentil", ["p99", "p95"], horizontal=True, key="ver_pct")

    do_loocv = predicoes[predicoes["pct"] == percentil][
        ["method", "codigo_estacao", "predicted", "observed"]
    ]
    do_loocv = do_loocv[do_loocv["method"].str.startswith("V")]
    v5 = v5_por_estacao(percentil)
    todas = pd.concat([do_loocv, v5[v5["method"] == "V5 (LOOCV real)"]], ignore_index=True)

    versoes_loocv = sorted(todas["method"].unique())
    escolhidas = coluna_a.multiselect(
        "Versões", versoes_loocv, default=versoes_loocv, key="ver_metodos"
    )
    recorte = todas[todas["method"].isin(escolhidas)]
    if recorte.empty:
        st.info("Selecione ao menos uma versão.")
    else:
        grafico = px.scatter(
            recorte,
            x="observed",
            y="predicted",
            color="method",
            hover_name="codigo_estacao",
            opacity=0.55,
            height=520,
            labels={"observed": f"Observado na estação — {percentil} (m/s)",
                    "predicted": f"Predito pela versão — {percentil} (m/s)",
                    "method": ""},
        )
        limite = [
            float(min(recorte["observed"].min(), recorte["predicted"].min())),
            float(max(recorte["observed"].max(), recorte["predicted"].max())),
        ]
        grafico.add_shape(type="line", x0=limite[0], y0=limite[0], x1=limite[1], y1=limite[1],
                          line=dict(dash="dash", color="#888", width=1))
        grafico.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        grafico.update_xaxes(range=limite)
        grafico.update_yaxes(range=limite)
        st.plotly_chart(grafico, width="stretch", config={"responsive": True})
        st.caption(
            "Cada ponto é uma estação, predita sem que ela participasse do próprio cálculo. "
            "A nuvem inteira abaixo da diagonal é o viés de subestimação das três versões."
        )

    st.caption(
        "A V1 não aparece: é legado e saiu do comparador. Todas as demais são preditas sem que a "
        "estação participe do próprio cálculo."
    )

    st.markdown("#### A V5 e o tamanho do vazamento")
    st.markdown(
        "A V5 é a **melhor da linhagem** — mas há duas leituras dela, e a diferença entre as duas "
        "é grande o bastante para mudar qualquer conclusão."
    )
    leituras = v5_por_estacao(percentil)
    linhas = []
    for rotulo in ["V5 (amostrada na produção)", "V5 (LOOCV real)"]:
        resumo = erro_resumido(leituras[leituras["method"] == rotulo])
        linhas.append({
            "Leitura": rotulo,
            "Viés": round(resumo["bias"], 2),
            "RMSE": round(resumo["rmse"], 2),
            "MAE": round(resumo["mae"], 2),
            "Corr": round(resumo["corr"], 2),
            "n": resumo["n"],
        })
    tabela_v5 = pd.DataFrame(linhas)
    coluna_esq, coluna_dir = st.columns([2, 3])
    coluna_esq.dataframe(tabela_v5, hide_index=True, width="stretch")
    diferenca = abs(tabela_v5.loc[1, "Viés"] - tabela_v5.loc[0, "Viés"])
    coluna_dir.warning(
        f"A grade de produção é amostrada na estação **usando a própria estação** no sistema de "
        f"kriging — ela se prevê em parte a si mesma. Medido em {percentil}, isso vale "
        f"**{diferenca:.2f} m/s de viés**. Só a linha de baixo é comparável com os demais métodos.",
        icon="⚠️",
    )

    st.divider()
    st.markdown("#### Mapas de comparação entre versões")
    figura("03_comparacao/v1_vs_v2_maps.png", "V1 e V2 lado a lado.")
    figura("03_comparacao/v3_diffs_p99.png", "O que a V3 mudou em relação à referência, no p99.")


# ---------------------------------------------------------------- teto

elif secao_atual == "teto":
    cabecalho_da_secao("teto")

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
            "F-madograma sobre a série diária das estações, é de **~10 km** (R²≈0,98). A V3 rodou "
            "com σ=2,0°, da ordem de **200 km** — cerca de 20× mais larga. Quanto mais larga a "
            "janela, mais vizinhos discordantes entram na média e mais o extremo se dilui.",
            icon="📏",
        )

    st.divider()
    st.markdown("#### O efeito local")
    figura(
        "03_comparacao/dome_200km_p99.png",
        "Recorte de 200 km ao redor de uma estação (★). Sem suavização a correção fica concentrada; "
        "IDW e kernel gaussiano espalham o efeito. Os dois últimos painéis mostram o que cada um mudou.",
    )

    st.markdown("#### O trade-off da largura")
    figura(
        "03_comparacao/tune_sigma_grid.png",
        "Varredura da largura do kernel. À esquerda, detalhe local e transições abruptas. "
        "À direita, o campo achata e a informação da estação se dissolve.",
    )


# ---------------------------------------------------------------- LOOCV

elif secao_atual == "loocv":
    cabecalho_da_secao("loocv")

    st.markdown(
        "**Validação cruzada leave-one-station-out:** cada estação é removida, seu valor é predito "
        "usando só as demais, e comparado ao observado. É a única régua que mede o que importa — "
        "acertar onde *não* há medição."
    )

    metricas = carregar("loocv_consolidated_metrics.csv")
    percentil = st.radio("Percentil", ["p99", "p95"], horizontal=True, key="loocv_pct")
    recorte = metricas[metricas["pct"] == percentil].copy()

    # A V5 não está na tabela consolidada (entrou na numeração depois). Calculada
    # aqui do mesmo cache, contra o mesmo observado — o método de cálculo foi
    # conferido reproduzindo uma linha já publicada da tabela.
    leituras = v5_por_estacao(percentil)
    resumo_v5 = erro_resumido(leituras[leituras["method"] == "V5 (LOOCV real)"])
    recorte = pd.concat([
        recorte,
        pd.DataFrame([{
            "method": "V5 (Kriging Ordinário)", "pct": percentil,
            "bias": resumo_v5["bias"], "rmse": resumo_v5["rmse"],
            "mae": resumo_v5["mae"], "corr": resumo_v5["corr"], "n": resumo_v5["n"],
        }]),
    ], ignore_index=True)
    recorte = recorte.sort_values("rmse").reset_index(drop=True)

    esquerda, direita = st.columns([3, 2])
    with esquerda:
        grafico = px.bar(
            recorte.sort_values("rmse", ascending=False),
            x="rmse", y="method", orientation="h",
            color="bias", color_continuous_scale="RdBu", color_continuous_midpoint=0,
            height=430,
            labels={"rmse": "RMSE (m/s) — menor é melhor", "method": "", "bias": "Viés"},
        )
        grafico.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(grafico, width="stretch", config={"responsive": True})
    with direita:
        exibir = recorte[["method", "bias", "rmse", "mae", "corr", "n"]].copy()
        exibir.columns = ["Método", "Viés", "RMSE", "MAE", "Corr", "n"]
        for coluna in ["Viés", "RMSE", "MAE", "Corr"]:
            exibir[coluna] = exibir[coluna].round(2)
        exibir["n"] = exibir["n"].astype(int)
        st.dataframe(exibir, hide_index=True, width="stretch")
        st.caption(
            "Todos em m/s, exceto correlação. A coluna `n` traz o número de estações de cada "
            "método — a V5 cobre um conjunto ligeiramente menor."
        )

    producao = recorte[recorte["method"].str.startswith(("V2", "V3", "V4", "V5"))]
    if not producao.empty:
        melhor_v = producao.sort_values("rmse").iloc[0]
        st.error(
            f"**As versões da linhagem ocupam o fim da lista.** V2 a V5 aparecem com viés de "
            f"{producao['bias'].min():.1f} a {producao['bias'].max():.1f} m/s — subestimando o "
            f"extremo por larga margem, exatamente o que a seção 3 prevê. A melhor delas é a "
            f"**{melhor_v['method']}** (RMSE {melhor_v['rmse']:.2f}), ainda assim atrás de todos "
            "os métodos alternativos.",
            icon="⚠️",
        )
    melhor = recorte.iloc[0]
    st.success(
        f"**{melhor['method']}** lidera entre os métodos rápidos (RMSE {melhor['rmse']:.2f} m/s, "
        f"viés {melhor['bias']:+.2f}). Vale notar: é uma variante do próprio IDW, com o expoente "
        "calibrado localmente em vez de fixo — parte do problema da V2 era o parâmetro, não o mecanismo.",
        icon="✅",
    )

    st.divider()
    st.markdown("#### Simulação condicional (MSP2) — melhor resultado, em subamostra de 62 estações")
    msp2 = carregar("msp2_loocv_fmadograma_p99.csv")
    erro = msp2["predicted"] - msp2["observed"]
    colunas = st.columns(3)
    colunas[0].metric("Viés", f"{erro.mean():+.2f} m/s")
    colunas[1].metric("RMSE", f"{(erro**2).mean()**0.5:.2f} m/s")
    colunas[2].metric("Correlação", f"{msp2['predicted'].corr(msp2['observed']):.2f}")
    dispersao = px.scatter(
        msp2, x="observed", y="predicted", hover_name="codigo_estacao", height=400,
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


# ---------------------------------------------------------------- mapas

elif secao_atual == "mapas":
    cabecalho_da_secao("mapas")

    st.markdown(
        "A seção 4 diz quem **prevê** melhor. Esta diz o que cada método **produz no mapa** — "
        "onde ele corrige, quanto, e com que textura espacial."
    )

    coluna_a, coluna_b = st.columns([3, 1])
    rotulo = coluna_a.selectbox("Método", list(MAPAS_025), key="mapa_metodo")
    percentil = coluna_b.radio("Percentil", ["p99", "p95"], horizontal=True, key="mapa_pct")
    chave = MAPAS_025[rotulo]
    numero = percentil[1:]

    campo = campo_mascarado(f"campo_025__{chave}__p{numero}", "025")
    referencia = campo_mascarado(f"campo_025__v2__p{numero}", "025")

    if campo is None:
        st.info(f"O campo de **{rotulo}** não foi pré-processado neste recorte.")
    else:
        esquerda, direita = st.columns(2)
        with esquerda:
            st.plotly_chart(
                mapa(campo, "025", f"{rotulo} — {percentil}"),
                width="stretch", config={"responsive": True},
            )
        with direita:
            if chave == "v2":
                st.info(
                    "A V2 é a própria referência das diferenças — não há painel de comparação "
                    "para ela. Escolha outro método para ver o que muda.",
                    icon="ℹ️",
                )
            elif referencia is None:
                st.info("Campo de referência indisponível.")
            else:
                st.plotly_chart(
                    mapa(campo - referencia, "025", f"{rotulo} − V2", divergente=True),
                    width="stretch", config={"responsive": True},
                )
        st.caption(
            "Esquerda: campo interpolado do método, grade contínua de 0,25°, domínio de máximos "
            "anuais. Direita: **vermelho = o método prevê mais que a V2**, azul = prevê menos. "
            "A V2 é a referência por ser o baseline de produção — a pergunta é o que muda ao "
            "adotar outro método no lugar dela."
        )

    st.divider()
    st.markdown("#### A grade fina da V5 (0,1°)")
    st.caption(
        "A V5 é a única versão gerada no grid nativo de 0,1° — cerca de 6× mais fino que o grid "
        "de 0,25° usado por todo o resto. Aqui a V2 aparece reamostrada para o mesmo grid, para "
        "que a diferença seja comparável."
    )

    v5 = campo_mascarado(f"campo_xavier__v5__p{numero}", "xavier")
    v2_fino = campo_mascarado(f"campo_xavier__v2_on_xavier__p{numero}", "xavier")
    if v5 is None:
        st.info("Grade fina da V5 indisponível.")
    else:
        esquerda, direita = st.columns(2)
        with esquerda:
            st.plotly_chart(
                mapa(v5, "xavier", f"V5 (Kriging Ordinário) — {percentil}"),
                width="stretch", config={"responsive": True},
            )
        with direita:
            if v2_fino is None:
                st.info("V2 reamostrada indisponível.")
            else:
                st.plotly_chart(
                    mapa(v5 - v2_fino, "xavier", f"V5 − V2 — {percentil}", divergente=True),
                    width="stretch", config={"responsive": True},
                )
        st.caption(
            "Estes são os campos de produção, não a leitura de validação cruzada — ver a ressalva "
            "de vazamento na seção 2."
        )


# ---------------------------------------------------------------- destino

elif secao_atual == "destino":
    cabecalho_da_secao("destino")

    esquerda, direita = st.columns(2)
    with esquerda, st.container(border=True):
        st.markdown("##### A conclusão metodológica")
        st.markdown(
            "**A interpolação por combinação convexa tem teto.** Recalibrar o kernel (V4) ou trocar "
            "o peso (V3) não resolve, porque o limite é do mecanismo, não do parâmetro. Os métodos "
            "que se saíram melhor são justamente os que saem dessa família: mirar o quantil "
            "diretamente, adaptar o expoente localmente, ou simular o processo extremal."
        )
    with direita, st.container(border=True):
        st.markdown("##### A comparação em aberto")
        st.markdown(
            "Uma base de IA desenvolvida em paralelo está sendo comparada contra a interpolação em "
            "pé de igualdade: mesmo recorte espacial, mesmo período fora da amostra de treino, e "
            "validação cruzada real do lado da interpolação. **O resultado dessa comparação está "
            "em consolidação e não é reportado aqui.**"
        )

    st.caption(
        "A régua em pé de igualdade é deliberada: a versão de produção usa a rede nacional inteira "
        "e o histórico completo, vantagem de informação que a base de IA não tinha — por isso a "
        "comparação usa uma versão restrita ao mesmo domínio e período."
    )


# ---------------------------------------------------------------- acervo

elif secao_atual == "acervo":
    st.subheader("📚 Acervo")
    st.caption("Figuras produzidas pelo pipeline, inclusive as que não entraram no percurso.")

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
    grade_de_figuras(grupos[escolhido])
