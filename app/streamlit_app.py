"""Painel público da Base Vendaval — a trilha de interpolação, do dado observado ao teto do método."""

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

# Uma paleta só para o painel inteiro: o observado é sempre branco, a produção
# sempre azul, o alternativo verde, o alerta laranja.
COR_GT = "#ffffff"
COR_PRODUCAO = "#3987e5"
COR_ALTERNATIVO = "#26a69a"
COR_ALERTA = "#ef6c4d"
COR_NEUTRO = "#9e9e9e"
ALTURA_GRAFICO = 420
ALTURA_MAPA = 560

st.markdown(
    """
<style>
  .block-container { padding-top: 2.2rem; max-width: 1480px; }
  h1, h2, h3 { letter-spacing: -0.01em; }
  h4 { margin-top: 1.5rem; margin-bottom: .4rem; font-size: 1.05rem; opacity: .95; }
  .hero {
    border: 1px solid rgba(57,135,229,.35); border-left: 4px solid #3987e5;
    border-radius: 12px; padding: 1.4rem 1.6rem;
    background: linear-gradient(100deg, rgba(57,135,229,.11), rgba(57,135,229,.02));
  }
  .hero h1 { margin: 0 0 .45rem 0; font-size: 2.1rem; }
  .hero p { margin: 0; opacity: .88; font-size: 1.02rem; line-height: 1.55; }
  .passo { border-left: 3px solid #3987e5; padding: .15rem 0 .15rem .8rem; margin-bottom: .7rem; }
  .passo b { color: #6aa8f0; }
  .grupo-nav {
    font-size: .72rem; letter-spacing: .09em; text-transform: uppercase;
    opacity: .5; margin: 1rem 0 .1rem .2rem;
  }
  div[data-testid="stMetricValue"] { font-size: 1.45rem; }
  section[data-testid="stSidebar"] div[role="radiogroup"] { gap: .1rem; }
</style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- dados


@st.cache_data
def carregar(nome: str) -> pd.DataFrame:
    return pd.read_csv(DADOS / nome)


@st.cache_resource
def pacote_de_mapas():
    return dict(np.load(DADOS / "mapas_espaciais.npz", allow_pickle=False))


@st.cache_data
def proporcao(caminho: str) -> float:
    """Largura/altura lida do cabeçalho do PNG — sem decodificar a imagem."""
    with open(caminho, "rb") as arquivo:
        cabecalho = arquivo.read(33)
    largura, altura = struct.unpack(">II", cabecalho[16:24])
    return largura / altura


@st.cache_data
def v5_por_estacao(percentil: str) -> pd.DataFrame:
    """V5 nas estações, nas duas leituras, alinhada ao observado do LOOCV.

    `LOOCV real` exclui a estação-alvo do próprio sistema de kriging; a de
    produção é a grade materializada, amostrada na estação usando ela mesma.
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
        juntos = fonte.merge(observado, on="codigo_estacao").rename(columns={percentil: "predicted"})
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


# ---------------------------------------------------------------- visuais


def estilo(figura, altura: int = ALTURA_GRAFICO):
    figura.update_layout(
        height=altura,
        margin=dict(l=10, r=10, t=44, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return figura


def figura(nome: str, legenda: str = "") -> None:
    """Desenha a figura com a largura que o formato dela pede."""
    caminho = FIGURAS / nome
    if not caminho.exists():
        st.info(f"Figura ausente: `{nome}`.")
        return
    if proporcao(str(caminho)) < 1.0:  # retrato — conter, senão domina a tela
        _, meio, _ = st.columns([1, 2, 1])
        alvo = meio
    else:
        alvo = st.container()
    with alvo:
        st.image(str(caminho), caption=legenda or None, width="stretch")


def grade_de_figuras(arquivos) -> None:
    fila = []

    def descarregar(itens):
        for inicio in range(0, len(itens), 2):
            for coluna, arquivo in zip(st.columns(2), itens[inicio : inicio + 2]):
                coluna.image(str(arquivo), caption=arquivo.stem.replace("_", " "), width="stretch")

    for arquivo in arquivos:
        aspecto = proporcao(str(arquivo))
        if aspecto >= 2.2 or aspecto < 1.0:
            descarregar(fila)
            fila = []
            st.image(str(arquivo), caption=arquivo.stem.replace("_", " "), width="stretch")
        else:
            fila.append(arquivo)
    descarregar(fila)


def base_geografica(titulo: str):
    pacote = pacote_de_mapas()
    figura_mapa = go.Figure()
    figura_mapa.add_trace(
        go.Scatter(x=pacote["contorno_x"], y=pacote["contorno_y"], mode="lines",
                   line=dict(color="rgba(255,255,255,.45)", width=1),
                   hoverinfo="skip", showlegend=False)
    )
    figura_mapa.update_layout(
        title=titulo, height=ALTURA_MAPA, margin=dict(l=10, r=10, t=48, b=10),
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(title=None, showgrid=False, scaleanchor="x", scaleratio=1),
    )
    return figura_mapa


def camada_estacoes(figura_mapa, colorir: bool = False, percentil: str = "p99"):
    """Sobrepõe as estações do INMET — o dado de referência — a um mapa."""
    estacoes = carregar("inmet_observado.csv")
    if colorir:
        marcador = dict(size=7, color=estacoes[percentil], colorscale="Turbo",
                        line=dict(color="rgba(0,0,0,.6)", width=0.6),
                        colorbar=dict(title="m/s"))
    else:
        marcador = dict(size=4, color="rgba(255,255,255,.85)",
                        line=dict(color="rgba(0,0,0,.5)", width=0.5))
    figura_mapa.add_trace(
        go.Scatter(
            x=estacoes["longitude"], y=estacoes["latitude"], mode="markers",
            marker=marcador, name="Estações INMET",
            customdata=np.stack([estacoes["codigo_estacao"], estacoes[percentil]], axis=-1),
            hovertemplate="%{customdata[0]}<br>observado %{customdata[1]:.1f} m/s<extra></extra>",
            showlegend=False,
        )
    )
    return figura_mapa


def campo_mascarado(chave: str, sufixo: str):
    pacote = pacote_de_mapas()
    if chave not in pacote:
        return None
    campo = pacote[chave].astype("float64").copy()
    campo[~pacote[f"mask_{sufixo}"]] = np.nan
    return campo


def mapa(campo: np.ndarray, sufixo: str, titulo: str, divergente: bool = False):
    """Heatmap geográfico com 1° de longitude = 1° de latitude.

    A trava de proporção é segura porque a navegação é por estado: seção não
    selecionada não é montada, então nunca há container escondido sem largura.
    """
    pacote = pacote_de_mapas()
    validos = campo[~np.isnan(campo)]
    if divergente:
        limite = float(np.percentile(np.abs(validos), 95)) if validos.size else 1.0
        zmin, zmax, escala = -limite, limite, "RdBu_r"
    else:
        zmin = float(validos.min()) if validos.size else 0.0
        zmax = float(validos.max()) if validos.size else 1.0
        escala = "Turbo"
    figura_mapa = base_geografica(titulo)
    figura_mapa.add_trace(
        go.Heatmap(z=campo, x=pacote[f"lons_{sufixo}"], y=pacote[f"lats_{sufixo}"],
                   zmin=zmin, zmax=zmax, colorscale=escala,
                   hovertemplate="lon %{x:.2f}<br>lat %{y:.2f}<br>%{z:.2f} m/s<extra></extra>",
                   colorbar=dict(title="m/s"))
    )
    figura_mapa.data = figura_mapa.data[::-1]  # contorno por cima do campo
    return figura_mapa


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


# ---------------------------------------------------------------- navegação

SECOES = {
    "observado": ("1 · O dado observado", "O que o INMET mede, e por que ele é a régua de tudo."),
    "versoes": ("2 · O mecanismo e as versões", "Como se corrige a grade, e o que mudou de V1 a V5."),
    "teto": ("3 · O teto estrutural", "Por que nenhuma dessas versões reproduz um extremo local."),
    "loocv": ("4 · A régua", "Qual método prevê melhor onde não há estação."),
    "mapas": ("5 · Os mapas", "O que cada método produz no território."),
    "series": ("6 · No tempo", "Como cada fonte se comporta ano a ano, no mesmo lugar."),
    "destino": ("7 · Para onde foi", "A conclusão metodológica e o que segue em aberto."),
}
GRUPOS = {
    "A análise": ["observado", "versoes", "teto"],
    "As evidências": ["loocv", "mapas", "series"],
    "O fecho": ["destino"],
}

if "secao" not in st.session_state:
    st.session_state.secao = "inicio"


def ir_para(destino: str) -> None:
    """Navegação. Sempre como callback: `secao` alimenta os rádios da lateral,
    e o Streamlit proíbe escrever numa chave de widget já instanciado."""
    st.session_state.secao = destino
    for grupo, chaves in GRUPOS.items():
        st.session_state[f"nav_{grupo}"] = destino if destino in chaves else None


def _selecionou(grupo: str) -> None:
    escolhido = st.session_state[f"nav_{grupo}"]
    if escolhido:
        ir_para(escolhido)


for _grupo, _chaves in GRUPOS.items():
    _nome = f"nav_{_grupo}"
    if _nome not in st.session_state:
        st.session_state[_nome] = (
            st.session_state.secao if st.session_state.secao in _chaves else None
        )

with st.sidebar:
    st.markdown("### 🌬️ Base Vendaval")
    st.caption("Correção de rajada por interpolação")
    st.button("🏠 Início", key="nav_inicio", width="stretch",
              on_click=ir_para, args=("inicio",))
    for grupo, chaves in GRUPOS.items():
        st.markdown(f'<div class="grupo-nav">{grupo}</div>', unsafe_allow_html=True)
        st.radio(
            grupo, chaves, key=f"nav_{grupo}", label_visibility="collapsed",
            format_func=lambda c: SECOES[c][0], on_change=_selecionou, args=(grupo,),
        )
    st.markdown('<div class="grupo-nav">Material</div>', unsafe_allow_html=True)
    st.button("📚 Acervo", key="nav_acervo", width="stretch",
              on_click=ir_para, args=("acervo",))
    st.divider()
    st.caption("Expanda qualquer figura no ícone que aparece ao passar o mouse.")

secao_atual = st.session_state.secao


def abre_secao(chave: str) -> None:
    titulo, resumo = SECOES[chave]
    st.subheader(titulo)
    st.caption(resumo)


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
        "Este painel cobre a **trilha de interpolação** — como espalhar a correção das estações "
        "para a grade, e como as versões dessa etapa foram testadas entre si. "
        "A trilha de IA é um trabalho separado.",
        icon="🧭",
    )

    metricas = carregar("loocv_consolidated_metrics.csv")
    p99 = metricas[metricas["pct"] == "p99"]
    colunas = st.columns(4)
    colunas[0].metric("Estações de referência", "615", help="Rede INMET, 2000–2025")
    colunas[1].metric("Versões da correção", "5", help="V1 a V5, ordem de produção")
    colunas[2].metric("Métodos comparados", str(p99["method"].nunique() + 1))
    colunas[3].metric("Escala espacial medida", "~10 km", help="F-madograma, R²≈0,98")

    st.divider()
    for grupo, chaves in GRUPOS.items():
        st.markdown(f"#### {grupo}")
        for inicio in range(0, len(chaves), 3):
            linha = st.columns(3)
            for coluna, chave in zip(linha, chaves[inicio : inicio + 3]):
                titulo, resumo = SECOES[chave]
                with coluna, st.container(border=True):
                    st.markdown(f"**{titulo}**")
                    st.caption(resumo)
                    st.button("Abrir", key=f"ir_{chave}", width="stretch",
                              on_click=ir_para, args=(chave,))

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


# ---------------------------------------------------------------- 1. observado

elif secao_atual == "observado":
    abre_secao("observado")

    estacoes = carregar("inmet_observado.csv")
    anual = carregar("inmet_anual.csv")
    percentil = st.radio("Percentil observado", ["p99", "p95"], horizontal=True, key="gt_pct")

    esquerda, direita = st.columns([3, 2])
    with esquerda:
        st.plotly_chart(
            camada_estacoes(
                base_geografica(f"INMET — {percentil} da rajada máxima anual"),
                colorir=True, percentil=percentil,
            ),
            width="stretch", config={"responsive": True},
        )
    with direita:
        indicadores = st.columns(3)
        indicadores[0].metric("Estações", f"{len(estacoes)}")
        indicadores[1].metric("Dias-estação", f"{anual['dias'].sum()/1e6:.1f} M")
        indicadores[2].metric(
            percentil, f"{estacoes[percentil].min():.0f}–{estacoes[percentil].max():.0f}",
            help="Faixa observada entre estações, em m/s",
        )
        st.markdown(
            "**A cobertura é desigual, e é daí que nasce o problema.** A rede é densa no Sul e "
            "no Sudeste e rala no Norte e no interior. Onde há estação, o resíduo a corrigir é "
            "conhecido. Onde não há, ele precisa ser **inferido dos vizinhos** — e é essa "
            "inferência que separa uma versão da outra."
        )
        st.dataframe(
            estacoes.nlargest(6, percentil)[["codigo_estacao", percentil, "n_obs"]]
            .rename(columns={percentil: f"{percentil} (m/s)", "n_obs": "anos"}),
            hide_index=True, width="stretch",
        )
        st.caption("As seis estações com o extremo observado mais alto.")

    st.markdown("#### Como a rajada se distribui")
    distribuicao = carregar("inmet_distribuicao.csv")
    sazonal = carregar("inmet_sazonal.csv")

    esquerda, direita = st.columns(2)
    with esquerda:
        grafico = px.bar(
            distribuicao[distribuicao["dias"] > 0], x="limite_inferior", y="dias",
            labels={"limite_inferior": "rajada diária máxima (m/s)", "dias": "dias-estação"},
        )
        grafico.update_traces(marker_color=COR_GT, marker_line_width=0)
        grafico.update_yaxes(type="log")
        st.plotly_chart(
            estilo(grafico).update_layout(title="Distribuição das rajadas diárias (escala log)"),
            width="stretch", config={"responsive": True},
        )
        total = int(distribuicao["dias"].sum())
        acima = int(distribuicao[distribuicao["limite_inferior"] >= 25]["dias"].sum())
        st.caption(
            f"{total:,} dias-estação. Acima de 25 m/s há apenas {acima:,} — "
            f"{acima/total*100:.3f}% do total.".replace(",", ".")
        )
    with direita:
        por_mes = sazonal.groupby("mes").agg(
            media=("rajada_media", "mean"), maxima=("rajada_max", "max")
        ).reset_index()
        MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                 "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
        por_mes["nome"] = por_mes["mes"].map(lambda m: MESES[m - 1])
        grafico = go.Figure()
        grafico.add_trace(go.Bar(x=por_mes["nome"], y=por_mes["media"],
                                 name="média diária", marker_color=COR_PRODUCAO))
        grafico.add_trace(go.Scatter(x=por_mes["nome"], y=por_mes["maxima"],
                                     name="máximo absoluto", mode="lines+markers",
                                     line=dict(color=COR_ALERTA, width=2), yaxis="y2"))
        grafico.update_layout(
            title="Sazonalidade",
            yaxis=dict(title="média (m/s)"),
            yaxis2=dict(title="máximo (m/s)", overlaying="y", side="right", showgrid=False),
        )
        st.plotly_chart(estilo(grafico), width="stretch", config={"responsive": True})
        pico = por_mes.loc[por_mes["media"].idxmax()]
        vale = por_mes.loc[por_mes["media"].idxmin()]
        st.caption(
            f"A média sobe no fim do inverno e na primavera — pico em {pico['nome']} "
            f"({pico['media']:.1f} m/s) contra {vale['nome']} ({vale['media']:.1f})."
        )

    with st.expander("Cobertura da rede ao longo do tempo"):
        por_ano = anual.groupby("ano").agg(
            estacoes=("codigo_estacao", "nunique"), dias=("dias", "sum")
        ).reset_index()
        grafico = px.area(por_ano, x="ano", y="estacoes",
                          labels={"ano": "", "estacoes": "estações com dado"})
        grafico.update_traces(line_color=COR_PRODUCAO, fillcolor="rgba(57,135,229,.25)")
        st.plotly_chart(
            estilo(grafico, 320).update_layout(title="Estações reportando por ano"),
            width="stretch", config={"responsive": True},
        )
        st.caption(
            "A rede cresce ao longo da série. Isso importa na leitura de qualquer tendência: "
            "mais estações em anos recentes significa mais chance de captar um extremo raro, "
            "independentemente de o clima ter mudado."
        )


# ---------------------------------------------------------------- 2. versões

elif secao_atual == "versoes":
    abre_secao("versoes")

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
            "exclusivamente o passo 2.**"
        )
    with direita:
        figura(
            "01_exploracao/boxplot_gusts_comparison.png",
            "Por que corrigir: as medianas quase coincidem, mas a estação registra rajadas muito "
            "acima do teto que a reanálise alcança. O erro está na cauda.",
        )

    st.markdown("#### As cinco versões")
    versoes = [
        ("V1", "Magnitude apenas", "Legado",
         "Resíduo interpolado por IDW somado à magnitude bruta do ERA5, sem tratar direção."),
        ("V2", "IDW p=2, k=15", "Baseline de produção antigo",
         "IDW clássico (peso 1/d²) mais interpolação da direção. Com p=2 o peso cai rápido "
         "demais: o campo vira mosaico quase-constante em torno de cada estação."),
        ("V3", "Gaussiano σ=2,0°, k=15", "Tentativa de corrigir a V2",
         "Kernel gaussiano no lugar do IDW. Resolveu a transição abrupta, mas perdeu extremos — "
         "σ=2,0° é 13× o default do próprio código."),
        ("V4", "Brown-Resnick calibrado", "Kernel recalibrado",
         "Peso calibrado pela escala de dependência medida, com distância geodésica. Mantém a "
         "limitação estrutural."),
        ("V5", "Kriging Ordinário, grid 0,1°", "Candidato de produção",
         "Sistema linear com todas as estações simultaneamente, num grid ~6× mais fino. "
         "Sai da família de peso fixo."),
    ]
    for sigla, metodo, status, descricao in versoes:
        with st.container(border=True):
            topo, corpo = st.columns([1, 6])
            topo.markdown(f"### {sigla}")
            topo.caption(status)
            corpo.markdown(f"**{metodo}**  \n{descricao}")
    st.caption("A numeração é ordem de produção, não de qualidade — a seção 4 mostra a ordem real.")

    st.markdown("#### O que cada versão produz, estação a estação")
    predicoes = carregar("loocv_per_station_predictions.csv")
    coluna_a, coluna_b = st.columns([3, 1])
    percentil = coluna_b.radio("Percentil", ["p99", "p95"], horizontal=True, key="ver_pct")

    do_loocv = predicoes[predicoes["pct"] == percentil][
        ["method", "codigo_estacao", "predicted", "observed"]
    ]
    do_loocv = do_loocv[do_loocv["method"].str.startswith("V")]
    leituras = v5_por_estacao(percentil)
    todas = pd.concat([do_loocv, leituras[leituras["method"] == "V5 (LOOCV real)"]],
                      ignore_index=True)
    escolhidas = coluna_a.multiselect(
        "Versões", sorted(todas["method"].unique()),
        default=sorted(todas["method"].unique()), key="ver_metodos",
    )
    recorte = todas[todas["method"].isin(escolhidas)]

    if recorte.empty:
        st.info("Selecione ao menos uma versão.")
    else:
        grafico = px.scatter(
            recorte, x="observed", y="predicted", color="method",
            hover_name="codigo_estacao", opacity=0.5,
            labels={"observed": f"Observado no INMET — {percentil} (m/s)",
                    "predicted": f"Predito pela versão — {percentil} (m/s)", "method": ""},
        )
        limite = [float(min(recorte["observed"].min(), recorte["predicted"].min())),
                  float(max(recorte["observed"].max(), recorte["predicted"].max()))]
        grafico.add_shape(type="line", x0=limite[0], y0=limite[0], x1=limite[1], y1=limite[1],
                          line=dict(dash="dash", color=COR_GT, width=1))
        grafico.update_xaxes(range=limite)
        grafico.update_yaxes(range=limite)
        st.plotly_chart(estilo(grafico, 500), width="stretch", config={"responsive": True})
        st.caption(
            "Cada ponto é uma estação, predita sem que ela participasse do próprio cálculo. "
            "A nuvem inteira abaixo da diagonal é o viés de subestimação. A V1 não aparece: "
            "é legado e saiu do comparador."
        )

    with st.expander("⚠️ A V5 tem duas leituras, e a diferença entre elas é grande"):
        linhas = []
        for rotulo in ["V5 (amostrada na produção)", "V5 (LOOCV real)"]:
            resumo = erro_resumido(leituras[leituras["method"] == rotulo])
            linhas.append({"Leitura": rotulo, "Viés": round(resumo["bias"], 2),
                           "RMSE": round(resumo["rmse"], 2), "MAE": round(resumo["mae"], 2),
                           "Corr": round(resumo["corr"], 2), "n": resumo["n"]})
        tabela_v5 = pd.DataFrame(linhas)
        coluna_esq, coluna_dir = st.columns([2, 3])
        coluna_esq.dataframe(tabela_v5, hide_index=True, width="stretch")
        coluna_dir.warning(
            f"A grade de produção é amostrada na estação **usando a própria estação** no sistema "
            f"de kriging — ela se prevê em parte a si mesma. Em {percentil} isso vale "
            f"**{abs(tabela_v5.loc[1, 'Viés'] - tabela_v5.loc[0, 'Viés']):.2f} m/s de viés**. "
            "Só a linha de baixo é comparável com os demais métodos.",
            icon="⚠️",
        )


# ---------------------------------------------------------------- 3. teto

elif secao_atual == "teto":
    abre_secao("teto")

    esquerda, direita = st.columns([3, 2])
    with esquerda:
        st.markdown(
            """
As versões calculam o resíduo interpolado como **combinação convexa** dos vizinhos:

$$\\hat{r}(x) = \\sum_j w_j(x)\\, r_j, \\qquad w_j \\ge 0, \\quad \\sum_j w_j = 1$$

Toda combinação convexa obedece $\\min_j r_j \\le \\hat{r}(x) \\le \\max_j r_j$.

**O valor interpolado nunca supera o maior resíduo entre os vizinhos usados** — e fica
estritamente abaixo dele sempre que os vizinhos discordarem, que é o caso quando uma estação
captura um evento extremo e as vizinhas não.

Não é bug de implementação. É propriedade do mecanismo, e vale para qualquer peso.
            """
        )
    with direita:
        st.warning(
            "**A V3 agravou isso por parâmetro.** A escala de dependência espacial real, medida "
            "por F-madograma sobre a série diária, é de **~10 km** (R²≈0,98). A V3 rodou com "
            "σ=2,0°, da ordem de **200 km** — cerca de 20× mais larga. Quanto mais larga a "
            "janela, mais vizinhos discordantes entram na média e mais o extremo se dilui.",
            icon="📏",
        )

    figura(
        "03_comparacao/tune_sigma_grid.png",
        "Varredura da largura do kernel. À esquerda, detalhe local e transições abruptas. "
        "À direita, o campo achata e a informação da estação se dissolve.",
    )

    with st.expander("O mesmo efeito, ampliado em torno de uma estação"):
        figura(
            "03_comparacao/dome_200km_p99.png",
            "Recorte de 200 km ao redor de uma estação (★). Os dois últimos painéis mostram o "
            "que cada método de suavização mudou.",
        )


# ---------------------------------------------------------------- 4. régua

elif secao_atual == "loocv":
    abre_secao("loocv")

    st.markdown(
        "**Validação cruzada leave-one-station-out:** cada estação é removida, seu valor é predito "
        "usando só as demais, e comparado ao observado. É a régua que mede o que importa — "
        "acertar onde *não* há medição."
    )

    metricas = carregar("loocv_consolidated_metrics.csv")
    percentil = st.radio("Percentil", ["p99", "p95"], horizontal=True, key="loocv_pct")
    recorte = metricas[metricas["pct"] == percentil].copy()

    # A V5 não está na tabela consolidada (entrou na numeração depois). Calculada
    # do mesmo cache, contra o mesmo observado — o método de cálculo foi conferido
    # reproduzindo uma linha já publicada da tabela.
    resumo_v5 = erro_resumido(
        v5_por_estacao(percentil).query("method == 'V5 (LOOCV real)'")
    )
    recorte = pd.concat([recorte, pd.DataFrame([{
        "method": "V5 (Kriging Ordinário)", "pct": percentil, "bias": resumo_v5["bias"],
        "rmse": resumo_v5["rmse"], "mae": resumo_v5["mae"], "corr": resumo_v5["corr"],
        "n": resumo_v5["n"]}])], ignore_index=True).sort_values("rmse").reset_index(drop=True)

    esquerda, direita = st.columns([3, 2])
    with esquerda:
        grafico = px.bar(
            recorte.sort_values("rmse", ascending=False),
            x="rmse", y="method", orientation="h",
            color="bias", color_continuous_scale="RdBu", color_continuous_midpoint=0,
            labels={"rmse": "RMSE (m/s) — menor é melhor", "method": "", "bias": "Viés"},
        )
        st.plotly_chart(estilo(grafico, 440), width="stretch", config={"responsive": True})
    with direita:
        exibir = recorte[["method", "bias", "rmse", "mae", "corr", "n"]].copy()
        exibir.columns = ["Método", "Viés", "RMSE", "MAE", "Corr", "n"]
        for coluna in ["Viés", "RMSE", "MAE", "Corr"]:
            exibir[coluna] = exibir[coluna].round(2)
        exibir["n"] = exibir["n"].astype(int)
        st.dataframe(exibir, hide_index=True, width="stretch")
        st.caption("Todos em m/s, exceto correlação. `n` = estações avaliadas.")

    producao = recorte[recorte["method"].str.startswith(("V2", "V3", "V4", "V5"))]
    melhor = recorte.iloc[0]
    st.error(
        f"**As versões da linhagem ocupam o fim da lista.** V2 a V5 aparecem com viés de "
        f"{producao['bias'].min():.1f} a {producao['bias'].max():.1f} m/s. A melhor delas é a "
        f"**{producao.sort_values('rmse').iloc[0]['method']}** — ainda assim atrás de todos os "
        "métodos alternativos.",
        icon="⚠️",
    )
    st.success(
        f"**{melhor['method']}** lidera entre os métodos rápidos (RMSE {melhor['rmse']:.2f} m/s, "
        f"viés {melhor['bias']:+.2f}). É uma variante do próprio IDW, com o expoente calibrado "
        "localmente em vez de fixo — parte do problema da V2 era o parâmetro, não o mecanismo.",
        icon="✅",
    )

    with st.expander("Simulação condicional (MSP2) — melhor resultado, em subamostra de 62 estações"):
        msp2 = carregar("msp2_loocv_fmadograma_p99.csv")
        erro = msp2["predicted"] - msp2["observed"]
        colunas = st.columns(3)
        colunas[0].metric("Viés", f"{erro.mean():+.2f} m/s")
        colunas[1].metric("RMSE", f"{(erro**2).mean()**0.5:.2f} m/s")
        colunas[2].metric("Correlação", f"{msp2['predicted'].corr(msp2['observed']):.2f}")
        dispersao = px.scatter(msp2, x="observed", y="predicted", hover_name="codigo_estacao",
                               labels={"observed": "Observado (m/s)", "predicted": "Predito (m/s)"})
        limite = [float(min(msp2["observed"].min(), msp2["predicted"].min())),
                  float(max(msp2["observed"].max(), msp2["predicted"].max()))]
        dispersao.add_shape(type="line", x0=limite[0], y0=limite[0], x1=limite[1], y1=limite[1],
                            line=dict(dash="dash", color=COR_GT, width=1))
        st.plotly_chart(estilo(dispersao, 380), width="stretch", config={"responsive": True})
        st.caption(
            "Simulação Monte Carlo do processo max-estável completo. É o melhor resultado medido, "
            "mas custa minutos por ponto — serve como auditoria pontual, não para gerar a grade "
            "histórica inteira."
        )


# ---------------------------------------------------------------- 5. mapas

elif secao_atual == "mapas":
    abre_secao("mapas")

    st.markdown(
        "A seção anterior diz quem **prevê** melhor. Esta diz o que cada método **produz no mapa** — "
        "onde corrige, quanto, e com que textura espacial."
    )

    coluna_a, coluna_b, coluna_c = st.columns([3, 1, 1])
    rotulo = coluna_a.selectbox("Campo", list(MAPAS_025), key="mapa_metodo")
    percentil = coluna_b.radio("Percentil", ["p99", "p95"], horizontal=True, key="mapa_pct")
    sobrepor = coluna_c.toggle("Estações", value=True, key="mapa_estacoes",
                               help="Sobrepõe as estações do INMET ao campo")
    numero = percentil[1:]
    chave = MAPAS_025[rotulo]

    campo = campo_mascarado(f"campo_025__{chave}__p{numero}", "025")
    referencia = campo_mascarado(f"campo_025__v2__p{numero}", "025")

    if campo is None:
        st.info(f"O campo de **{rotulo}** não foi pré-processado neste recorte.")
    else:
        esquerda, direita = st.columns(2)
        with esquerda:
            figura_campo = mapa(campo, "025", f"{rotulo} — {percentil}")
            if sobrepor:
                camada_estacoes(figura_campo, percentil=percentil)
            st.plotly_chart(figura_campo, width="stretch", config={"responsive": True})
        with direita:
            if chave == "v2":
                st.info(
                    "A V2 é a própria referência das diferenças — não há painel de comparação "
                    "para ela. Escolha outro método para ver o que muda.",
                    icon="ℹ️",
                )
            else:
                figura_diferenca = mapa(campo - referencia, "025", f"{rotulo} − V2",
                                        divergente=True)
                if sobrepor:
                    camada_estacoes(figura_diferenca, percentil=percentil)
                st.plotly_chart(figura_diferenca, width="stretch", config={"responsive": True})
        st.caption(
            "Esquerda: campo interpolado, grade contínua de 0,25°, com as estações sobrepostas. "
            "Direita: **vermelho = prevê mais que a V2**, azul = prevê menos. A V2 é a referência "
            "por ser o baseline de produção."
        )

    with st.expander("A grade fina da V5 (0,1°, ~6× mais resolução)"):
        v5 = campo_mascarado(f"campo_xavier__v5__p{numero}", "xavier")
        v2_fino = campo_mascarado(f"campo_xavier__v2_on_xavier__p{numero}", "xavier")
        if v5 is None:
            st.info("Grade fina indisponível.")
        else:
            esquerda, direita = st.columns(2)
            with esquerda:
                figura_v5 = mapa(v5, "xavier", f"V5 — {percentil}")
                if sobrepor:
                    camada_estacoes(figura_v5, percentil=percentil)
                st.plotly_chart(figura_v5, width="stretch", config={"responsive": True})
            with direita:
                st.plotly_chart(
                    mapa(v5 - v2_fino, "xavier", f"V5 − V2 — {percentil}", divergente=True),
                    width="stretch", config={"responsive": True},
                )
            st.caption(
                "A V2 aparece reamostrada no mesmo grid, para a diferença ser comparável. "
                "São os campos de produção — ver a ressalva de vazamento na seção 2."
            )


# ---------------------------------------------------------------- 6. séries

elif secao_atual == "series":
    abre_secao("series")

    st.markdown(
        "As seções anteriores olham o espaço. Esta olha o **tempo**: dez estações da Região Sul, "
        "espalhadas por amostragem de ponto mais distante, com o máximo anual de cada fonte "
        "sobreposto."
    )

    pontos = carregar("pontos_fixos.csv")
    series = carregar("series_pontos_fixos.csv")
    ORDEM = ["INMET (real)", "Base Pré-Interpolada", "ERA5 original", "V2 (IDW p=2, k=15)",
             "V3 (Gaussiano σ=2.0, k=15)", "V5 (Kriging Ordinário, Xavier 0.1°)"]
    CORES = {"INMET (real)": COR_GT, "Base Pré-Interpolada": "#b39ddb",
             "ERA5 original": COR_NEUTRO, "V2 (IDW p=2, k=15)": COR_PRODUCAO,
             "V3 (Gaussiano σ=2.0, k=15)": COR_ALTERNATIVO,
             "V5 (Kriging Ordinário, Xavier 0.1°)": COR_ALERTA}

    disponiveis = [f for f in ORDEM if f in set(series["fonte"])]
    escolhidas = st.multiselect("Fontes", disponiveis, default=disponiveis, key="serie_fontes")
    recorte = series[series["fonte"].isin(escolhidas)]

    st.info(
        "**Como ler.** A linha branca é a observação. A **Base Pré-Interpolada** é "
        "`max(INMET, ERA5)` na própria estação, antes de qualquer interpolação — acompanha a "
        "observação por construção, e serve para separar o erro da *fórmula* do erro da "
        "*interpolação*. Quando uma versão se afasta da branca num ano de pico, é a interpolação "
        "diluindo o extremo.",
        icon="🧭",
    )

    for inicio in range(0, len(pontos), 2):
        linha = st.columns(2)
        for coluna, codigo in zip(linha, pontos["codigo_estacao"].tolist()[inicio : inicio + 2]):
            da_estacao = recorte[recorte["codigo_estacao"] == codigo]
            if da_estacao.empty:
                coluna.info(f"Sem dado para {codigo}.")
                continue
            grafico = px.line(
                da_estacao.sort_values("ano"), x="ano", y="rajada_ms", color="fonte",
                markers=True, color_discrete_map=CORES, category_orders={"fonte": ORDEM},
                labels={"ano": "", "rajada_ms": "rajada máx. (m/s)", "fonte": ""},
            )
            grafico.update_layout(
                height=330, title=dict(text=f"Estação {codigo}", font=dict(size=15)),
                margin=dict(l=10, r=10, t=42, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0, font=dict(size=10)),
            )
            grafico.update_traces(line=dict(width=3.2), marker=dict(size=8),
                                  selector=dict(name="INMET (real)"))
            coluna.plotly_chart(grafico, width="stretch", config={"responsive": True})

    with st.expander("Quais são os dez pontos, e por que estes"):
        st.caption(
            "Seleção por *farthest-point sampling*: parte da estação mais central e vai somando "
            "a que está mais longe do conjunto já escolhido, espalhando os pontos em vez de "
            "concentrá-los onde a rede é densa."
        )
        st.dataframe(pontos, hide_index=True, width="stretch")
    st.caption(
        "Só entram as fontes com grade diária materializada. V4, MSP, EVK, QK, RBF e IDW "
        "Adaptativo existem como campo de percentil e aparecem na seção 5."
    )


# ---------------------------------------------------------------- 7. destino

elif secao_atual == "destino":
    abre_secao("destino")

    esquerda, direita = st.columns(2)
    with esquerda, st.container(border=True):
        st.markdown("##### A conclusão metodológica")
        st.markdown(
            "**A interpolação por combinação convexa tem teto.** Recalibrar o kernel (V4) ou "
            "trocar o peso (V3) não resolve, porque o limite é do mecanismo, não do parâmetro. "
            "Os métodos que se saíram melhor são os que saem dessa família: mirar o quantil "
            "diretamente, adaptar o expoente localmente, ou simular o processo extremal."
        )
    with direita, st.container(border=True):
        st.markdown("##### A comparação em aberto")
        st.markdown(
            "Uma base de IA desenvolvida em paralelo está sendo comparada contra a interpolação "
            "em pé de igualdade: mesmo recorte espacial, mesmo período fora da amostra de treino, "
            "e validação cruzada real do lado da interpolação. **O resultado dessa comparação "
            "está em consolidação e não é reportado aqui.**"
        )
    st.caption(
        "A régua em pé de igualdade é deliberada: a versão de produção usa a rede nacional inteira "
        "e o histórico completo, vantagem de informação que a base de IA não tinha."
    )


# ---------------------------------------------------------------- acervo

elif secao_atual == "acervo":
    st.subheader("📚 Acervo")
    st.caption("Figuras produzidas pelo pipeline, inclusive as que não entraram no percurso.")

    grupos = {
        "Comparação entre versões e ajuste": sorted((FIGURAS / "03_comparacao").glob("*.png")),
        "Campos médios da reanálise": sorted((FIGURAS / "01_exploracao").glob("*_mean_map.png")),
        "Covariáveis por estação": sorted((FIGURAS / "01_exploracao").glob("map_covariate_*.png")),
        "Recorte de interesse": sorted((FIGURAS / "01_exploracao").glob("map_roi_*.png")),
        "Distribuições e correlação": sorted((FIGURAS / "01_exploracao").glob("boxplot_*.png"))
        + sorted((FIGURAS / "01_exploracao").glob("pdf_*.png"))
        + sorted((FIGURAS / "01_exploracao").glob("heatmap_*.png"))
        + sorted((FIGURAS / "01_exploracao").glob("inmet_*.png"))
        + sorted((FIGURAS / "01_exploracao").glob("map_station_*.png")),
    }
    grupos = {t: a for t, a in grupos.items() if a}
    escolhido = st.selectbox("Grupo", list(grupos),
                             format_func=lambda t: f"{t} ({len(grupos[t])})", key="acervo_grupo")
    grade_de_figuras(grupos[escolhido])
