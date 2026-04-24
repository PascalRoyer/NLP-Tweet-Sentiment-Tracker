"""
Classification de texte avec TF-IDF + scikit-learn.

Modes (--mode) :
  - trinary (défaut) : positive / neutral / negative — regroupe les émotions du CSV en 3 classes.
  - emotions : prédit la colonne Emotion telle quelle (multiclasse : neutral, anger, love, …).
  - sentiment : positif / négatif seulement (sans les neutres).
  - binary : CSV déjà annoté positive / negative.

Prédiction : --predict / --interactive — mode rapide par défaut ; ajoute --full pour l'ancien gros modèle (lent).

Évaluation (sans --predict) : découpage train / validation / test par défaut (--eval-split three-way,
~70%% / 15%% / 15%% avec --test-fraction 0.15 et --val-fraction 0.15).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_HANDLE_RE = re.compile(r"@\w+")
_MULTI_SPACE = re.compile(r"\s+")

# Mapping émotion -> sentiment (jeu type "Emotion" / tweets)
POSITIVE = {"happiness", "love", "fun", "relief", "enthusiasm", "surprise"}
NEGATIVE = {"hate", "anger", "worry", "sadness", "fear", "boredom", "empty"}

# Mots / formules de polarité (anglais) — complète le modèle quand le jeu "émotion" ne suffit pas
SENTIMENT_NEG = frozenset(
    """
    despise despises despised loathe loathes loathed detest detests detested abhor abhors abhorred
    hate hates hated disgusting disgusted disgust awful terrible horrible horrid vile nasty gross
    worst worse bad badly sucks suck sucked pathetic useless worthless trash rubbish shame shameful
    regret regrets regretted angry angrier furious rage outraged offended horrible nightmare depressed
    depressing depression sad sadly sorrow grief miserable pathetic hurt hurting hurts painful pain
    annoying annoyed frustrates frustrated frustration stupid idiot morons moron dumb hateable
    ugly nastily sickening repulsive revolting obnoxious contempt scorn resent resents bitter
    """.split()
)
SENTIMENT_POS = frozenset(
    """
    love loves loved adore adores adored amazing wonderful great excellent perfect best good better
    happily happiness happy glad gladly joy joyful enjoy enjoyed enjoying fantastic awesome brilliant
    beautiful lovely nice nicest thank thanks grateful pleased delight delightful success winner
    win wins winning proud proudly hope hopeful bless blessed superb incredible fabulous cherish
    """.split()
)
# Comptent double (forte polarité)
INTENSE_NEG = frozenset({"despise", "despises", "despised", "loathe", "hate", "hates", "abhor", "detest"})
INTENSE_POS = frozenset({"love", "loves", "adore", "perfect", "wonderful", "amazing", "excellent"})
NEG_SUBPHRASES = (
    "can't stand",
    "cannot stand",
    "sick of",
    "fed up",
    "don't like",
    "do not like",
    "not good",
    "not happy",
    "pisses me off",
)
POS_SUBPHRASES = (
    "thank you",
    "thanks so",
    "feel great",
    "so happy",
    "love it",
    "love this",
)

LEXICON_BLEND_WEIGHT = 0.5  # part du lexique quand au moins un signal lexique est présent

# Menaces / violence / suicide (anglais) — le jeu est ~80 % "neutral", le ML sous-estime souvent hate/anger.
_HARM_PHRASES = (
    "kill you",
    "kill him",
    "kill her",
    "kill them",
    "kill myself",
    "want to kill",
    "wanna kill",
    "gonna kill",
    "going to kill",
    "murder you",
    "hurt you",
    "stab you",
    "shoot you",
    "strangle you",
    "beat you to death",
    "wish you were dead",
    "end your life",
    "take your life",
    "hang myself",
    "commit suicide",
)
_HARM_WORD_RE = re.compile(
    r"\b(kill|kills|killed|killing|murder|murdered|stab|stabbing|shoot|shooting|strangle)\b",
    re.I,
)


def detect_harmful_threat_language(text: str) -> bool:
    t = text.lower().replace("\u2019", "'")
    if any(p in t for p in _HARM_PHRASES):
        return True
    if _HARM_WORD_RE.search(t):
        return True
    if re.search(r"\bkys\b", t):
        return True
    return False


def pick_threat_emotion(classes: list[str]) -> str | None:
    """Émotion du CSV la plus adaptée aux menaces (si présente dans les classes)."""
    for cand in ("hate", "anger", "worry", "sadness", "empty"):
        if cand in classes:
            return cand
    return None


# Si le modele met une emotion "positive" ou neutre sur une menace, on affiche une correction.
_EMOTION_OK_FOR_THREAT = frozenset({"hate", "anger", "worry", "sadness"})


def preprocess_text(text: str) -> str:
    """Allège le bruit type tweet / URL (le sac de mots s'en sort mieux)."""
    t = str(text)
    t = _URL_RE.sub(" ", t)
    t = _HANDLE_RE.sub(" ", t)
    t = _MULTI_SPACE.sub(" ", t).strip()
    return t


def load_emotion_csv(path: Path, max_rows: int | None, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=max_rows)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Colonnes attendues : {text_col!r}, {label_col!r}. Trouvé : {list(df.columns)}")
    df = df[[text_col, label_col]].dropna()
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    df[text_col] = df[text_col].astype(str).map(preprocess_text)
    return df


def emotions_to_sentiment(labels: pd.Series) -> pd.Series:
    def map_one(x: str) -> str | None:
        if x in POSITIVE:
            return "positif"
        if x in NEGATIVE:
            return "negatif"
        return None

    return labels.map(map_one)


def emotions_to_trinary(labels: pd.Series) -> pd.Series:
    """Mappe les étiquettes Emotion du CSV vers positive / neutral / negative (anglais)."""

    def map_one(x: str) -> str:
        s = str(x).strip().lower()
        if s in POSITIVE:
            return "positive"
        if s in NEGATIVE:
            return "negative"
        if s == "neutral":
            return "neutral"
        return "neutral"

    return labels.map(map_one)


TRINARY_CM_LABELS = ["negative", "neutral", "positive"]


def _word_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        max_features=100_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.92,
        sublinear_tf=True,
    )


def _char_tfidf() -> TfidfVectorizer:
    # Mieux sur fautes, abréviations et mots rares (tweets courts)
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=35_000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=False,
    )


def build_feature_union() -> FeatureUnion:
    return FeatureUnion([("word", _word_tfidf()), ("char", _char_tfidf())])


def _logistic() -> LogisticRegression:
    return LogisticRegression(
        solver="saga",
        max_iter=2500,
        C=2.0,
        class_weight="balanced",
        random_state=42,
    )


def build_pipeline_tfidf_logreg() -> Pipeline:
    return Pipeline([("features", build_feature_union()), ("clf", _logistic())])


def build_pipeline_tfidf_nb() -> Pipeline:
    return Pipeline([("features", build_feature_union()), ("clf", MultinomialNB(alpha=0.05))])


def build_pipeline_ensemble() -> Pipeline:
    # n_jobs=1 : sous Windows, n_jobs=-1 + joblib peut provoquer des erreurs "processus introuvable"
    # si l'utilisateur interrompt (Ctrl+C) ou sur certaines machines.
    return Pipeline(
        [
            ("features", build_feature_union()),
            (
                "clf",
                VotingClassifier(
                    estimators=[("lr", _logistic()), ("nb", MultinomialNB(alpha=0.05))],
                    voting="soft",
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_pipeline_fast_predict() -> Pipeline:
    """TF-IDF mots seulement + LR — entraînement beaucoup plus rapide (recommandé pour --predict)."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    max_features=40_000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    max_iter=2000,
                    C=1.0,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _print_metrics_block(
    y_true,
    y_pred,
    title: str,
    *,
    report_style: str,
) -> None:
    """report_style: 'binary' | 'trinary' | 'multiclass'."""
    print(title)
    print(classification_report(y_true, y_pred, digits=4))
    if report_style == "binary":
        cm = confusion_matrix(y_true, y_pred, labels=["negatif", "positif"])
        print("Matrice de confusion [negatif, positif] (lignes=vérité, colonnes=pred) :")
    elif report_style == "trinary":
        cm = confusion_matrix(y_true, y_pred, labels=TRINARY_CM_LABELS)
        print("Matrice de confusion [negative, neutral, positive] (lignes=vérité, colonnes=pred) :")
    else:
        cm = confusion_matrix(y_true, y_pred)
        print("Matrice de confusion (lignes=vérité, colonnes=pred) :")
    print(cm)


def run_experiment(
    X_train,
    X_test,
    y_train,
    y_test,
    name: str,
    pipeline: Pipeline,
    *,
    report_style: str,
    test_title: str = "Jeu de TEST",
) -> None:
    """Un seul jeu d'évaluation (ancien comportement train / test)."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f"\n=== {name} ===")
    _print_metrics_block(
        y_test,
        y_pred,
        f"--- {test_title} ({len(y_test)} exemples) ---",
        report_style=report_style,
    )


def run_experiment_train_val_test(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    name: str,
    pipeline: Pipeline,
    *,
    report_style: str,
) -> None:
    """
    Entraîne sur train uniquement ; évalue sur validation puis sur test.
    Le jeu de test sert à l'évaluation finale (ne pas le réutiliser pour ajuster les hyperparamètres).
    """
    pipeline.fit(X_train, y_train)
    print(f"\n=== {name} ===")
    y_pred_val = pipeline.predict(X_val)
    _print_metrics_block(
        y_val,
        y_pred_val,
        f"--- VALIDATION ({len(y_val)} ex.) — suivi / réglage des hyperparamètres ---",
        report_style=report_style,
    )
    y_pred_test = pipeline.predict(X_test)
    _print_metrics_block(
        y_test,
        y_pred_test,
        f"--- TEST final ({len(y_test)} ex.) — évaluation sans biais, à ne consulter qu'une fois ---",
        report_style=report_style,
    )


def predict_emotion_texts(
    pipeline: Pipeline,
    texts: list[str],
    *,
    label_col: str = "Emotion",
    top_k: int = 5,
    harm_adjust: bool = True,
) -> None:
    """Prédit les mêmes étiquettes que la colonne du CSV (ex. neutral, anger, love, hate…)."""
    if not texts:
        return
    classes = list(pipeline.named_steps["clf"].classes_)
    probas = pipeline.predict_proba(texts)
    for text, row in zip(texts, probas, strict=True):
        scored = sorted(zip(classes, row, strict=True), key=lambda x: -x[1])
        best_e, best_p = scored[0]
        pr_map = {e: float(p) for e, p in scored}
        print(f"Texte: {text!r}")

        harm = harm_adjust and detect_harmful_threat_language(text)
        alt = pick_threat_emotion(classes) if harm else None
        model_wrong_for_threat = bool(
            harm and alt and best_e not in _EMOTION_OK_FOR_THREAT
        )
        if model_wrong_for_threat:
            p_alt = pr_map.get(alt, 0.0)
            print(
                f"  => {label_col} (corrige menace/violence - le ML se trompe souvent ici): "
                f"{alt!r}  (score brut modele pour {alt!r}: {p_alt:.3f})"
            )
            print(
                f"  => {label_col} (score brut du modele seul): {best_e!r}  prob={best_p:.3f} "
                "(peut etre neutral, fun, etc. - peu fiable sur menaces)"
            )
        else:
            print(f"  => {label_col} (comme dans le CSV): {best_e!r}  prob={best_p:.3f}")
        head = scored[: min(top_k, len(scored))]
        detail = ", ".join(f"{e!r}={float(p):.3f}" for e, p in head)
        print(f"  top-{len(head)} du modele: {detail}")


def print_prediction(label: str, proba: dict[str, float], note: str = "") -> None:
    p_pos = proba.get("positif", 0.0)
    p_neg = proba.get("negatif", 0.0)
    extra = f" {note}" if note else ""
    print(f"  => {label}  (P(negatif)={p_neg:.3f}, P(positif)={p_pos:.3f}){extra}")


def _lexicon_masses(text: str) -> tuple[float, float]:
    """Retourne (masse_negative, masse_positive) pour mélange avec le modèle."""
    t = text.lower().replace("\u2019", "'")
    words = re.findall(r"[a-z']+", t)
    neg_m = 0.0
    pos_m = 0.0
    for w in words:
        if w in SENTIMENT_NEG:
            neg_m += 2.0 if w in INTENSE_NEG else 1.0
        if w in SENTIMENT_POS:
            pos_m += 2.0 if w in INTENSE_POS else 1.0
    for p in NEG_SUBPHRASES:
        if p in t:
            neg_m += 1.5
    for p in POS_SUBPHRASES:
        if p in t:
            pos_m += 1.5
    return neg_m, pos_m


def blend_model_with_lexicon(model_pr: dict[str, float], text: str) -> tuple[dict[str, float], str]:
    """Combine les probas sklearn avec un signal lexique si des mots forts sont présents."""
    neg_m, pos_m = _lexicon_masses(text)
    if neg_m <= 0 and pos_m <= 0:
        return model_pr, ""

    total = neg_m + pos_m
    p_neg_lex = neg_m / total
    p_pos_lex = pos_m / total
    w = LEXICON_BLEND_WEIGHT
    p_neg = (1.0 - w) * model_pr.get("negatif", 0.0) + w * p_neg_lex
    p_pos = (1.0 - w) * model_pr.get("positif", 0.0) + w * p_pos_lex
    s = p_neg + p_pos
    if s > 0:
        p_neg /= s
        p_pos /= s
    return {"negatif": p_neg, "positif": p_pos}, "[lexique+modele]"


def predict_texts(pipeline: Pipeline, texts: list[str], use_lexicon_blend: bool) -> None:
    if not texts:
        return
    classes = list(pipeline.named_steps["clf"].classes_)
    probas = pipeline.predict_proba(texts)
    for text, row in zip(texts, probas, strict=True):
        pr_model = {classes[i]: float(row[i]) for i in range(len(classes))}
        if use_lexicon_blend:
            pr, note = blend_model_with_lexicon(pr_model, text)
            label = "negatif" if pr["negatif"] >= pr["positif"] else "positif"
        else:
            pr, note = pr_model, ""
            idx = max(range(len(row)), key=lambda i: row[i])
            label = classes[idx]
        print(f"Texte: {text!r}")
        print_prediction(label, pr, note)


def _pipeline_for_classifier(name: str) -> Pipeline:
    if name == "logreg":
        return build_pipeline_tfidf_logreg()
    if name == "nb":
        return build_pipeline_tfidf_nb()
    return build_pipeline_ensemble()


def run_predict_mode(
    X,
    y,
    classifier: str,
    predict_one: str | None,
    interactive: bool,
    use_lexicon_blend: bool,
    *,
    predict_emotions: bool,
    label_col: str,
    extra_texts: list[str] | None = None,
    fast_model: bool = True,
    harm_adjust: bool = True,
) -> None:
    """Entraîne sur tout le jeu (pour usage perso) puis classe les textes saisis."""
    if classifier == "both":
        print("Pour --predict / --interactive, utilisation du mode vote souple (ensemble).", file=sys.stderr)
        classifier = "ensemble"

    if fast_model:
        pipeline = build_pipeline_fast_predict()
        print(
            "Mode rapide : TF-IDF (mots) + regression logistique (sans n-grams caracteres ni ensemble).",
            flush=True,
        )
    else:
        pipeline = _pipeline_for_classifier(classifier)
        print(
            "Mode complet : TF-IDF mots + caracteres + classificateur choisi (plus lent, plus de RAM).",
            flush=True,
        )
    print(
        "Entraînement du modèle sur toutes les données étiquetées…",
        flush=True,
    )
    pipeline.fit(X, y)
    print("Entraînement termine.", flush=True)
    if predict_emotions:
        print(
            f"Sortie = valeurs de la colonne {label_col!r} (même vocabulaire que le CSV, en minuscules).\n",
            flush=True,
        )
    elif use_lexicon_blend:
        print(
            "(Prédictions : mélange TF-IDF + petit lexique anglais pour les mots forts type 'despise'.)\n",
            flush=True,
        )

    def dispatch(texts: list[str]) -> None:
        if predict_emotions:
            predict_emotion_texts(
                pipeline,
                texts,
                label_col=label_col,
                harm_adjust=harm_adjust,
            )
        else:
            predict_texts(pipeline, texts, use_lexicon_blend)

    if predict_one is not None:
        dispatch([preprocess_text(predict_one)])

    if extra_texts:
        for t in extra_texts:
            dispatch([t])

    if interactive:
        print(
            "\n"
            + "=" * 60
            + "\nPRET : le curseur attend ton texte en dessous.\n"
            "Tape un tweet ou une phrase, puis Entree. Ligne vide = quitter.\n"
            + "=" * 60
            + "\n",
            flush=True,
        )
        while True:
            try:
                line = input("tweet> ").strip()
            except EOFError:
                break
            if not line:
                break
            dispatch([preprocess_text(line)])


def normalize_binary_labels(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    pos = {"positif", "positive", "pos", "1", "true", "yes"}
    neg = {"negatif", "négatif", "negative", "neg", "0", "false", "no"}
    out = []
    for v in s:
        if v in pos:
            out.append("positif")
        elif v in neg:
            out.append("negatif")
        else:
            out.append(None)
    return pd.Series(out, index=series.index)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TF-IDF + LR / Naive Bayes / ensemble — émotions (défaut) ou sentiment binaire"
    )
    parser.add_argument(
        "--mode",
        choices=("emotions", "sentiment", "binary", "emotion"),
        default="emotions",
        help="emotions: prédire la colonne Emotion (recommandé pour ce CSV) | sentiment: positif/négatif "
        "depuis les émotions | binary: CSV déjà pos/neg | emotion: alias de sentiment (ancien nom)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "Sentiment Projet",
        help="Chemin vers le CSV (colonnes text + Emotion par défaut)",
    )
    parser.add_argument("--text-col", default="text", help="Nom colonne texte")
    parser.add_argument(
        "--label-col",
        default="Emotion",
        help="emotions/sentiment/emotion: colonne Emotion | binary: colonne pos/neg",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200_000,
        help="Nombre max de lignes lues (0 = tout le fichier ; plus = souvent un peu mieux, plus de RAM)",
    )
    parser.add_argument(
        "--classifier",
        choices=("logreg", "nb", "ensemble", "both"),
        default="ensemble",
        help="logreg / nb / ensemble (vote souple, défaut) / both = entraîne et compare les 3 (évaluation)",
    )
    parser.add_argument(
        "--eval-split",
        choices=("three-way", "two-way"),
        default="three-way",
        help="three-way: train / validation / test (~70%% / 15%% / 15%% par défaut) | two-way: train / test seulement",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Fraction du jeu réservée au TEST final (three-way uniquement, défaut 0.15)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction du jeu réservée à la VALIDATION (three-way, défaut 0.15 ; train = le reste)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Mode two-way uniquement : proportion du jeu de test (défaut 0.2)",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        metavar="TEXT",
        help="Classer un seul texte puis quitter (mode rapide par défaut ; --full pour modèle lourd)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Mode interactif : saisir des tweets ligne par ligne (dans le terminal intégré, pas via Run ▶)",
    )
    parser.add_argument(
        "--tweet-file",
        type=Path,
        default=None,
        metavar="FICHIER",
        help="Un tweet par ligne (UTF-8) ; classifie chaque ligne puis quitte (utile si input() ne marche pas)",
    )
    parser.add_argument(
        "--no-lexicon-blend",
        action="store_true",
        help="Désactive le mélange avec le lexique (TF-IDF seul, peut se tromper sur 'despise', etc.)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Avec --predict / --interactive / --tweet-file : utilise le gros modele (mots+chars, ensemble selon --classifier). Beaucoup plus lent.",
    )
    parser.add_argument(
        "--no-harm-adjust",
        action="store_true",
        help="Ne pas corriger les menaces/violence (sinon, si le modele dit 'neutral', on propose hate/anger selon mots-cles)",
    )
    args = parser.parse_args()

    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else None
    if not args.data.is_file():
        print(f"Fichier introuvable : {args.data}", file=sys.stderr)
        return 1

    mode = "sentiment" if args.mode == "emotion" else args.mode

    print("Chargement des données…", flush=True)
    df = load_emotion_csv(args.data, max_rows, args.text_col, args.label_col)
    predict_emotions = mode == "emotions"
    binary_sentiment = mode == "sentiment"

    if mode == "emotions":
        df = df[df[args.label_col].astype(str).str.strip() != ""]
        X = df[args.text_col].values
        y = df[args.label_col].values
    elif mode == "sentiment":
        df["sentiment"] = emotions_to_sentiment(df[args.label_col])
        df = df.dropna(subset=["sentiment"])
        y = df["sentiment"].values
        X = df[args.text_col].values
    else:
        df["sentiment"] = normalize_binary_labels(df[args.label_col])
        df = df.dropna(subset=["sentiment"])
        y = df["sentiment"].values
        X = df[args.text_col].values

    if len(df) < 500:
        print("Trop peu d'exemples. Vérifiez les colonnes ou --max-rows.", file=sys.stderr)
        return 1

    print(f"Échantillon utilisable : {len(df)} phrases.")
    print(f"Répartition des classes : {pd.Series(y).value_counts().to_dict()}")
    if predict_emotions:
        print(f"Cible d'apprentissage : colonne {args.label_col!r} (étiquettes du CSV, ex. neutral, love, anger…).")

    if args.interactive and not sys.stdin.isatty():
        print(
            "\nImpossible de saisir du texte : ce n'est pas un terminal interactif.\n"
            "Dans Cursor : ouvre le Terminal intégré (Ctrl+` ou Terminal > New Terminal), puis lance :\n"
            f'  python "{Path(__file__).resolve()}" --interactive --max-rows 25000\n'
            "Ne lance pas le script avec le bouton ▶ Run : il n'accepte pas le clavier.\n"
            'Autres options : --predict "ton tweet"   ou   --tweet-file tweets.txt\n',
            file=sys.stderr,
            flush=True,
        )
        return 2

    if args.tweet_file is not None:
        if not args.tweet_file.is_file():
            print(f"Fichier introuvable : {args.tweet_file}", file=sys.stderr)
            return 1
        raw = args.tweet_file.read_text(encoding="utf-8", errors="replace").splitlines()
        lines = [preprocess_text(line) for line in raw if preprocess_text(line)]
        if not lines:
            print("Aucune ligne non vide dans le fichier.", file=sys.stderr)
            return 1
        run_predict_mode(
            X,
            y,
            args.classifier,
            predict_one=None,
            interactive=False,
            use_lexicon_blend=not args.no_lexicon_blend and not predict_emotions,
            predict_emotions=predict_emotions,
            label_col=args.label_col,
            extra_texts=lines,
            fast_model=not args.full,
            harm_adjust=not args.no_harm_adjust,
        )
        return 0

    if args.predict is not None or args.interactive:
        run_predict_mode(
            X,
            y,
            args.classifier,
            args.predict,
            args.interactive,
            use_lexicon_blend=not args.no_lexicon_blend and not predict_emotions,
            predict_emotions=predict_emotions,
            label_col=args.label_col,
            extra_texts=None,
            fast_model=not args.full,
            harm_adjust=not args.no_harm_adjust,
        )
        return 0

    tf = args.test_fraction
    vf = args.val_fraction
    if args.eval_split == "three-way":
        if not (0 < tf < 1 and 0 < vf < 1 and tf + vf < 1):
            print(
                "test-fraction + val-fraction doit etre < 1 et chaque fraction dans ]0,1[.",
                file=sys.stderr,
            )
            return 1
        val_size_within_trainval = vf / (1.0 - tf)
        try:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=tf, random_state=args.random_state, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp,
                y_temp,
                test_size=val_size_within_trainval,
                random_state=args.random_state,
                stratify=y_temp,
            )
        except ValueError:
            print(
                "stratify impossible (classe trop rare ?). Découpe sans stratification.",
                file=sys.stderr,
            )
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=tf, random_state=args.random_state, stratify=None
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp,
                y_temp,
                test_size=val_size_within_trainval,
                random_state=args.random_state,
                stratify=None,
            )
        n = len(X)
        print(
            f"\nDécoupe three-way : train={len(y_train)} ({100*len(y_train)/n:.1f}%), "
            f"validation={len(y_val)} ({100*len(y_val)/n:.1f}%), "
            f"test={len(y_test)} ({100*len(y_test)/n:.1f}%).",
            flush=True,
        )
        bs = mode == "sentiment" or mode == "binary"
        if args.classifier in ("logreg", "both"):
            run_experiment_train_val_test(
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                "TF-IDF mots+chars + Régression logistique",
                build_pipeline_tfidf_logreg(),
                binary_sentiment=bs,
            )
        if args.classifier in ("nb", "both"):
            run_experiment_train_val_test(
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                "TF-IDF mots+chars + Naive Bayes",
                build_pipeline_tfidf_nb(),
                binary_sentiment=bs,
            )
        if args.classifier in ("ensemble", "both"):
            run_experiment_train_val_test(
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                "TF-IDF mots+chars + vote souple (LR + NB)",
                build_pipeline_ensemble(),
                binary_sentiment=bs,
            )
        return 0

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )
    except ValueError:
        print(
            "stratify impossible (classe trop rare ?). Découpe sans stratification.",
            file=sys.stderr,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=None
        )

    n = len(X)
    print(
        f"\nDécoupe two-way : train={len(y_train)} ({100*len(y_train)/n:.1f}%), "
        f"test={len(y_test)} ({100*len(y_test)/n:.1f}%).",
        flush=True,
    )
    bs = mode == "sentiment" or mode == "binary"
    if args.classifier in ("logreg", "both"):
        run_experiment(
            X_train,
            X_test,
            y_train,
            y_test,
            "TF-IDF mots+chars + Régression logistique",
            build_pipeline_tfidf_logreg(),
            binary_sentiment=bs,
        )
    if args.classifier in ("nb", "both"):
        run_experiment(
            X_train,
            X_test,
            y_train,
            y_test,
            "TF-IDF mots+chars + Naive Bayes",
            build_pipeline_tfidf_nb(),
            binary_sentiment=bs,
        )
    if args.classifier in ("ensemble", "both"):
        run_experiment(
            X_train,
            X_test,
            y_train,
            y_test,
            "TF-IDF mots+chars + vote souple (LR + NB)",
            build_pipeline_ensemble(),
            binary_sentiment=bs,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
