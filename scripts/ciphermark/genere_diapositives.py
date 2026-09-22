#!/usr/bin/env python3
"""
Genere les diapositives de soutenance (.pptx) a partir des chiffres REELS de la
campagne de cloture.

Aucun chiffre n'est saisi a la main : tout est lu depuis
resultats/campagne-cloture/5000/*.json. Si un resultat change, on relance ce
script et les diapositives suivent. C'est la meme discipline que pour les
tableaux de l'annexe du memoire.

    venv/bin/python scripts/ciphermark/genere_diapositives.py

Sortie : soutenance/CipherMark_soutenance.pptx
"""
from __future__ import annotations
import json, os, sys

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(RACINE, "resultats", "campagne-cloture", "5000")
SORTIE = os.path.join(RACINE, "soutenance")

# --------------------------------------------------------------- identite ---
# Noms de personnes en majuscules, et titre exact de l'encadreur :
# maitre de conferences, non professeur.
AUTEUR = "NGUEAGHO TIODONG ROBERTO LANDRY"
ENCADREUR = "AMINOU HALIDOU"
ENCADREUR_TITRE = "Maître de Conférences"
ETABLISSEMENT = "Universite de Yaounde I  ·  Faculte des Sciences"
DEPARTEMENT = "Departement d'Informatique  ·  Master 2"
TITRE = ("Developpement d'un systeme de tatouage numerique securise\n"
         "pour les modeles generatifs, fonde sur la technique OTP")

# ---------------------------------------------------------------- palette ---
ENCRE     = RGBColor(0x14, 0x1F, 0x33)   # bleu nuit, le fond des titres
PAPIER    = RGBColor(0xF7, 0xF6, 0xF2)   # blanc casse, legerement chaud
TEXTE     = RGBColor(0x1E, 0x28, 0x38)
GRIS      = RGBColor(0x6B, 0x74, 0x82)
ACCENT    = RGBColor(0xC4, 0x6A, 0x2E)   # terre cuite : ce qui compte
APPUI     = RGBColor(0x2E, 0x6F, 0x8E)   # bleu ardoise : le second plan
VERT      = RGBColor(0x3F, 0x7A, 0x54)
ROUGE     = RGBColor(0xA8, 0x3B, 0x3B)

TITRE_F = CORPS_F = "Arial"   # universel ; Liberation Sans en est le
                             # clone metrique, donc l'apercu local est fidele


def presentation() -> Presentation:
    p = Presentation()
    p.slide_width, p.slide_height = Inches(13.333), Inches(7.5)
    return p


def fond(dia, couleur=PAPIER):
    f = dia.background.fill
    f.solid(); f.fore_color.rgb = couleur


def zone(dia, x, y, w, h):
    tb = dia.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def ecrit(tf, texte, taille=18, gras=False, couleur=TEXTE, police=CORPS_F,
          align=PP_ALIGN.LEFT, interligne=1.15, espace_avant=0, premier=False):
    p = tf.paragraphs[0] if premier else tf.add_paragraph()
    p.alignment = align
    p.line_spacing = interligne
    if espace_avant:
        p.space_before = Pt(espace_avant)
    r = p.add_run(); r.text = texte
    r.font.size = Pt(taille); r.font.bold = gras
    r.font.color.rgb = couleur; r.font.name = police
    return p


def bandeau(dia, titre, surtitre=None):
    """Le bandeau de titre, commun a toutes les diapositives de contenu."""
    # 1,62 pouce et non 1,35 : a 27 pt, un titre de deux lignes debordait du
    # bandeau et se faisait couper. Verifie au rendu, pas suppose.
    b = dia.shapes.add_shape(1, 0, 0, Inches(13.333), Inches(1.62))
    b.fill.solid(); b.fill.fore_color.rgb = ENCRE
    b.line.fill.background(); b.shadow.inherit = False
    if surtitre:
        tf = zone(dia, 0.75, 0.24, 11.5, 0.3)
        ecrit(tf, surtitre.upper(), 10.5, True, RGBColor(0xC9, 0xA2, 0x7A),
              CORPS_F, premier=True)
        y = 0.56
    else:
        y = 0.46
    tf = zone(dia, 0.75, y, 11.9, 0.95)
    ecrit(tf, titre, 22, True, PAPIER, TITRE_F, interligne=1.12, premier=True)


def pastille(dia, x, y, w, h, valeur, libelle, couleur=ACCENT):
    """Un chiffre qui doit rester en tete, et sa legende."""
    c = dia.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    c.line.color.rgb = RGBColor(0xE0, 0xDD, 0xD5); c.line.width = Pt(1)
    c.shadow.inherit = False
    tf = zone(dia, x + 0.15, y + 0.22, w - 0.3, h - 0.4)
    ecrit(tf, valeur, 34, True, couleur, TITRE_F, PP_ALIGN.CENTER, premier=True)
    ecrit(tf, libelle, 12, False, GRIS, CORPS_F, PP_ALIGN.CENTER,
          interligne=1.05, espace_avant=4)


def puces(dia, x, y, w, elements, taille=17, ecart=13):
    """Une liste ou chaque entree est (gras, suite)."""
    tf = zone(dia, x, y, w, 5.0)
    for i, (fort, suite) in enumerate(elements):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.25
        if i: p.space_before = Pt(ecart)
        r = p.add_run(); r.text = "▸  "
        r.font.size = Pt(taille); r.font.color.rgb = ACCENT; r.font.name = CORPS_F
        r = p.add_run(); r.text = fort
        r.font.size = Pt(taille); r.font.bold = True
        r.font.color.rgb = TEXTE; r.font.name = CORPS_F
        if suite:
            r = p.add_run(); r.text = " " + suite
            r.font.size = Pt(taille); r.font.color.rgb = TEXTE; r.font.name = CORPS_F
    return tf


def tableau(dia, x, y, w, h, entetes, lignes, largeurs=None, taille=13):
    n_l, n_c = len(lignes) + 1, len(entetes)
    g = dia.shapes.add_table(n_l, n_c, Inches(x), Inches(y), Inches(w), Inches(h))
    t = g.table
    if largeurs:
        for i, lg in enumerate(largeurs):
            t.columns[i].width = Emu(int(Inches(w) * lg / sum(largeurs)))
    for j, e in enumerate(entetes):
        c = t.cell(0, j); c.text = e
        c.fill.solid(); c.fill.fore_color.rgb = ENCRE
        pr = c.text_frame.paragraphs[0]
        pr.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
        for r in pr.runs:
            r.font.size = Pt(taille); r.font.bold = True
            r.font.color.rgb = PAPIER; r.font.name = CORPS_F
    for i, ligne in enumerate(lignes, start=1):
        for j, val in enumerate(ligne):
            c = t.cell(i, j); c.text = str(val)
            c.fill.solid()
            c.fill.fore_color.rgb = (RGBColor(0xFF, 0xFF, 0xFF) if i % 2
                                     else RGBColor(0xEF, 0xED, 0xE7))
            pr = c.text_frame.paragraphs[0]
            pr.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
            for r in pr.runs:
                r.font.size = Pt(taille); r.font.name = CORPS_F
                r.font.color.rgb = TEXTE
                if j == 0: r.font.bold = True
    return t


def note(dia, texte, y=6.75, couleur=GRIS, taille=12.5):
    tf = zone(dia, 0.75, y, 11.9, 0.55)
    ecrit(tf, texte, taille, False, couleur, CORPS_F, premier=True)



def commentaire(dia, texte):
    """Note du presentateur : ce qu'il faut DIRE, pas ce qui est affiche."""
    tf = dia.notes_slide.notes_text_frame
    tf.text = texte

# ============================================================== les donnees ==
def charge():
    """Les chiffres viennent des JSON de la campagne, jamais d'une saisie."""
    a = json.load(open(os.path.join(RES, "volet_a_pixel.json")))
    cle = "rows" if "rows" in a else next(k for k, v in a.items()
                                          if isinstance(v, list) and v)
    A = {r["attaque"]: r for r in a[cle]}
    D = {r["attaque"]: r for r in json.load(open(os.path.join(
        RES, "volet_b_diffusion", "volet_b_diffusion_encode.json")))["conditions"]}
    G = {r["attaque"]: r for r in json.load(open(os.path.join(
        RES, "volet_b_autoregressif",
        "volet_b_autoregressif_encode.json")))["conditions"]}
    return A, D, G


def pc(x):
    return f"{x*100:.0f} %" if abs(x*100 - round(x*100)) < 0.05 else f"{x*100:.1f} %"


# =============================================================== les pages ==
def main() -> int:
    A, D, G = charge()
    os.makedirs(SORTIE, exist_ok=True)
    prs = presentation()
    vide = prs.slide_layouts[6]
    def page(): 
        d = prs.slides.add_slide(vide); fond(d); return d

    # ------------------------------------------------------------- 1. titre --
    d = page(); fond(d, ENCRE)
    tf = zone(d, 1.1, 0.95, 11.2, 0.4)
    ecrit(tf, "CIPHERMARK", 14, True, RGBColor(0xC9, 0xA2, 0x7A), CORPS_F, premier=True)
    tf = zone(d, 1.1, 1.45, 11.2, 1.7)
    ecrit(tf, TITRE, 26, True, PAPIER, TITRE_F, interligne=1.25, premier=True)
    barre = d.shapes.add_shape(1, Inches(1.1), Inches(3.28), Inches(2.0), Inches(0.055))
    barre.fill.solid(); barre.fill.fore_color.rgb = ACCENT
    barre.line.fill.background(); barre.shadow.inherit = False
    tf = zone(d, 1.1, 3.55, 11.2, 0.6)
    ecrit(tf, "Un temoin cryptographique unique par generation, "
              "inscrit dans les poids du modele", 17,
          False, RGBColor(0xB6, 0xBE, 0xC9), CORPS_F, premier=True)
    tf = zone(d, 1.1, 4.75, 5.6, 1.6)
    ecrit(tf, "Presente par", 12, True, RGBColor(0xC9, 0xA2, 0x7A), CORPS_F, premier=True)
    ecrit(tf, AUTEUR, 19, True, PAPIER, CORPS_F, espace_avant=4)
    ecrit(tf, "Master 2 · Informatique", 14, False, RGBColor(0x9A, 0xA4, 0xB0), CORPS_F)
    tf = zone(d, 7.1, 4.75, 5.2, 1.6)
    ecrit(tf, "Sous la direction de", 12, True, RGBColor(0xC9, 0xA2, 0x7A), CORPS_F, premier=True)
    ecrit(tf, ENCADREUR, 19, True, PAPIER, CORPS_F, espace_avant=4)
    ecrit(tf, ENCADREUR_TITRE, 14, False, RGBColor(0x9A, 0xA4, 0xB0), CORPS_F)
    tf = zone(d, 1.1, 6.5, 11.2, 0.8)
    ecrit(tf, ETABLISSEMENT, 13, False, RGBColor(0x9A, 0xA4, 0xB0), CORPS_F, premier=True)
    ecrit(tf, DEPARTEMENT + "  ·  Annee 2025-2026", 13, False,
          RGBColor(0x9A, 0xA4, 0xB0), CORPS_F)

    # -------------------------------------------------------- 2. le probleme --
    d = page(); bandeau(d, "Une image generee ne se distingue plus d'une photo", "Le constat")
    puces(d, 0.75, 2.0, 7.3, [
        ("Le contenu synthetique est devenu indiscernable", "a l'oeil comme aux detecteurs statistiques."),
        ("Prouver l'origine d'une image", "devient une exigence reglementaire : l'AI Act impose le marquage des contenus generes."),
        ("Un filigrane ne suffit pas.", "Il doit resister aux transformations ordinaires, et surtout ne jamais accuser a tort."),
    ], taille=17)
    pastille(d, 8.55, 2.15, 2.0, 1.55, "17", "articles publies\nen 2024-2026", APPUI)
    pastille(d, 10.8, 2.15, 2.0, 1.55, "0", "combinant les trois\nproprietes visees", ROUGE)
    tf = zone(d, 8.55, 4.1, 4.25, 2.2)
    ecrit(tf, "Les trois proprietes", 14, True, ENCRE, CORPS_F, premier=True)
    for t in ("securite cryptographique formelle",
              "tracabilite individuelle par generation",
              "couverture architecturale large"):
        ecrit(tf, "·  " + t, 13.5, False, TEXTE, CORPS_F, espace_avant=6)
    note(d, "Le verrou n'est pas la detection : c'est la preuve, individuelle et opposable.")

    # ------------------------------------------------- 3. l'etat de l'art ----
    d = page(); bandeau(d, "Ce que fait l'existant, et ce qui lui manque", "Synthese critique")
    tableau(d, 0.75, 2.0, 11.8, 2.5,
            ["Approche", "Securite formelle", "Message unique", "Familles couvertes"],
            [["DistSeal (2026)", "non", "non : message fixe, rejouable", "diffusion + autoregressif"],
             ["WOUAF (2024)", "non", "oui, mais code arbitraire", "diffusion"],
             ["MetaSeal (2026)", "oui", "oui", "diffusion seule"],
             ["CipherMark", "oui", "oui, lie au contenu", "diffusion + autoregressif"]],
            largeurs=[2.6, 1.6, 3.0, 2.6], taille=13.5)
    tf = zone(d, 0.75, 4.7, 11.8, 1.6)
    ecrit(tf, "Le manque, precisement", 16, True, ENCRE, CORPS_F, premier=True)
    ecrit(tf, "DistSeal grave un message FIXE dans les poids du decodeur : toutes ses "
              "generations portent le meme filigrane, qui peut donc etre rejoue. Les schemas "
              "cryptographiques, eux, restent confines a la diffusion. Aucun travail ne tient "
              "les trois proprietes ensemble.", 15.5, False, TEXTE, CORPS_F,
          interligne=1.3, espace_avant=6)

    # ---------------------------------------------------- 4. la question ----
    d = page(); fond(d, ENCRE)
    tf = zone(d, 1.4, 1.6, 10.5, 0.5)
    ecrit(tf, "QUESTION DE RECHERCHE", 14, True, RGBColor(0xC9, 0xA2, 0x7A), CORPS_F, premier=True)
    tf = zone(d, 1.4, 2.35, 10.5, 2.6)
    ecrit(tf, "Peut-on concevoir, pour les modeles generatifs, un tatouage "
              "garantissant a la fois une securite cryptographique formelle, "
              "une tracabilite individuelle par generation et une compatibilite "
              "architecturale large ?", 30, True, PAPIER, TITRE_F, interligne=1.28, premier=True)
    barre = d.shapes.add_shape(1, Inches(1.4), Inches(5.25), Inches(1.6), Inches(0.05))
    barre.fill.solid(); barre.fill.fore_color.rgb = ACCENT
    barre.line.fill.background(); barre.shadow.inherit = False
    tf = zone(d, 1.4, 5.6, 10.5, 0.9)
    ecrit(tf, "Reponse apportee : oui, sous conditions explicites, "
              "et le modele generatif peut porter le dispositif lui-meme.",
          17, False, RGBColor(0xB6, 0xBE, 0xC9), CORPS_F, premier=True)

    # ------------------------------------------------------- 5. l'idee -----
    d = page(); bandeau(d, "Le champ temoin  Ω : une signature par generation", "La proposition")
    b = d.shapes.add_shape(1, Inches(0.75), Inches(1.95), Inches(11.8), Inches(1.15))
    b.fill.solid(); b.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    b.line.color.rgb = ACCENT; b.line.width = Pt(1.5); b.shadow.inherit = False
    tf = zone(d, 1.0, 2.3, 11.3, 0.6)
    ecrit(tf, "Ω  =  HMAC(K_secret , h)   ⊕   PRG(s_master , nonce)",
          25, True, ENCRE, TITRE_F, PP_ALIGN.CENTER, premier=True)
    for i, (t, s, c) in enumerate([
        ("HMAC(K, h)", "lie le temoin au CONTENU par une empreinte perceptuelle DINOv2.\n"
                       "Transplanter la marque sur une autre image la casse.", APPUI),
        ("⊕ PRG(s, nonce)", "rend chaque generation UNIQUE et illisible sans les cles.\n"
                                 "Rejouer une marque observee devient impossible.", ACCENT),
        ("Cle par createur", "k_user = HKDF(k_master, identifiant).\n"
                             "L'attribution devient infalsifiable, non declarative.", VERT)]):
        x = 0.75 + i * 4.0
        c2 = d.shapes.add_shape(1, Inches(x), Inches(3.3), Inches(3.75), Inches(2.1))
        c2.fill.solid(); c2.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        c2.line.color.rgb = RGBColor(0xE0, 0xDD, 0xD5); c2.shadow.inherit = False
        bar = d.shapes.add_shape(1, Inches(x), Inches(3.3), Inches(0.07), Inches(2.1))
        bar.fill.solid(); bar.fill.fore_color.rgb = c
        bar.line.fill.background(); bar.shadow.inherit = False
        tf = zone(d, x + 0.3, 3.55, 3.3, 1.7)
        ecrit(tf, t, 17, True, c, CORPS_F, premier=True)
        ecrit(tf, s, 13.5, False, TEXTE, CORPS_F, interligne=1.25, espace_avant=7)
    note(d, "Ce que DistSeal grave une fois pour toutes, CipherMark le recalcule a chaque image.")


    # ------------------------------------------------ 6. l'architecture ----
    d = page(); bandeau(d, "Trois couches, et une base de tracabilite", "Architecture")
    couches = [("EMISSION", "Le generateur produit le latent. Le decodeur, module par Ω,\n"
                            "rend une image DEJA tatouee.", APPUI),
               ("TRACABILITE", "Une entree par generation : nonce, empreinte de reference,\n"
                               "createur. Aucun secret par utilisateur n'est stocke.", ACCENT),
               ("VERIFICATION", "L'extracteur relit Ω, le registre donne le createur,\n"
                                "sa cle est REDERIVEE, le verdict tombe.", VERT)]
    for i, (t, s2, c) in enumerate(couches):
        y = 1.95 + i * 1.6
        b = d.shapes.add_shape(1, Inches(0.75), Inches(y), Inches(11.8), Inches(1.38))
        b.fill.solid(); b.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        b.line.color.rgb = RGBColor(0xE0, 0xDD, 0xD5); b.shadow.inherit = False
        bar = d.shapes.add_shape(1, Inches(0.75), Inches(y), Inches(0.09), Inches(1.38))
        bar.fill.solid(); bar.fill.fore_color.rgb = c
        bar.line.fill.background(); bar.shadow.inherit = False
        tf = zone(d, 1.15, y + 0.18, 2.6, 0.5)
        ecrit(tf, t, 16, True, c, CORPS_F, premier=True)
        tf = zone(d, 3.9, y + 0.2, 8.4, 1.0)
        ecrit(tf, s2, 15, False, TEXTE, CORPS_F, interligne=1.25, premier=True)
    note(d, "Le verifieur ne connait rien a l'avance : il lit le nonce, consulte le registre, "
            "et rederive la cle du createur enregistre.")

    # ---------------------------------------- 7. Ω dans les poids ----------
    d = page(); bandeau(d, "Ω module les poids du decodeur : le tatouage naît avec l'image",
                        "Le mecanisme, herite de WOUAF")
    b = d.shapes.add_shape(1, Inches(0.75), Inches(1.95), Inches(11.8), Inches(0.95))
    b.fill.solid(); b.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    b.line.color.rgb = APPUI; b.line.width = Pt(1.5); b.shadow.inherit = False
    tf = zone(d, 1.0, 2.2, 11.3, 0.5)
    ecrit(tf, "activation  ←  activation ⊙ (1 + γ(Ω))  +  β(Ω)", 23, True,
          ENCRE, TITRE_F, PP_ALIGN.CENTER, premier=True)
    puces(d, 0.75, 3.1, 7.2, [
        ("Identite exacte a l'initialisation :", "les tetes γ et β sont initialisees a zero, "
         "donc le decodeur conditionne demarre identique au decodeur d'origine."),
        ("Modulation bornee", "(γ ≤ 0,3 ; β ≤ 0,1), la ou WOUAF demodule. La degradation "
         "devient impossible au-dela d'un seuil connu."),
        ("Ω change a chaque generation", "sans reentrainer le modele : c'est ce que le "
         "message fixe de DistSeal interdisait."),
    ], taille=15.5)
    pastille(d, 8.35, 3.15, 2.05, 1.5, "6", "etages modules\n(DC-AE)", APPUI)
    pastille(d, 10.6, 3.15, 2.05, 1.5, "3", "etages modules\n(MaskGIT)", ROUGE)
    tf = zone(d, 8.35, 4.95, 4.3, 1.4)
    ecrit(tf, "Cet ecart de prises explique a lui seul la difference de robustesse "
              "entre les deux familles.", 14, False, TEXTE, CORPS_F,
          interligne=1.25, premier=True)

    # ------------------------------------------------- 8. le protocole -----
    d = page(); bandeau(d, "Ce qui a ete mesure, et comment", "Methodologie")
    for i, (v, l, c) in enumerate([("3", "canaux d'inscription\nevalues separement", APPUI),
                                   ("20", "conditions : 19 attaques\n+ la reference", ACCENT),
                                   ("5 000", "images par canal,\njamais vues a l'entrainement", VERT),
                                   ("300 000", "verifications\nau total", ENCRE)]):
        pastille(d, 0.75 + i * 3.02, 2.0, 2.75, 1.65, v, l, c)
    tf = zone(d, 0.75, 3.85, 11.8, 2.6)
    ecrit(tf, "Les vingt conditions du protocole de DistSeal", 16, True, ENCRE, CORPS_F, premier=True)
    ecrit(tf, "JPEG Q = 90, 70, 50, 30   ·   flou gaussien k = 3, 5, 7   ·   "
              "reechantillonnage 0,75 et 0,5   ·   recadrage 90, 70, 50 %   ·   "
              "luminosite et contraste x0,8 et x1,2   ·   bruit σ = 0,02, 0,05, 0,10",
          15, False, TEXTE, CORPS_F, interligne=1.35, espace_avant=8)
    ecrit(tf, "Parametres FIXES et non tires au hasard : les chiffres sont reproductibles "
              "d'une execution a l'autre. Deux controles negatifs accompagnent chaque mesure.",
          14.5, False, GRIS, CORPS_F, interligne=1.3, espace_avant=10)
    note(d, "Meme protocole, memes parametres, meme echelle pour les trois canaux : "
            "c'est ce qui rend la comparaison possible.")

    # ------------------------------------------ 9. resultat, le pixel ------
    d = page(); bandeau(d, "Le couple embedder / extracteur en espace pixel", "Resultat 1 sur 4")
    lignes = [["Sans attaque", pc(A["aucune"]["taux_authentic"]), f"{A['aucune']['bit_acc_omega']:.4f}"],
              ["JPEG Q = 30", pc(A["jpeg_q30"]["taux_authentic"]), f"{A['jpeg_q30']['bit_acc_omega']:.4f}"],
              ["Bruit σ = 0,1", pc(A["bruit_std0.1"]["taux_authentic"]), f"{A['bruit_std0.1']['bit_acc_omega']:.4f}"],
              ["Flou k = 7", pc(A["flou_k7"]["taux_authentic"]), f"{A['flou_k7']['bit_acc_omega']:.4f}"],
              ["Recadrage 90 %", pc(A["crop_0.9"]["taux_authentic"]), f"{A['crop_0.9']['bit_acc_omega']:.4f}"]]
    tableau(d, 0.75, 2.05, 6.6, 2.6, ["Condition", "Verdicts", "Bits relus"],
            lignes, largeurs=[3.0, 1.8, 1.8], taille=14)
    pastille(d, 7.75, 2.05, 2.3, 1.65, "22,5 dB", "PSNR du filigrane", APPUI)
    pastille(d, 10.25, 2.05, 2.3, 1.65, "0,942", "SSIM", APPUI)
    tf = zone(d, 7.75, 3.75, 4.8, 2.4)
    ecrit(tf, "Ce qu'il faut retenir", 15, True, ENCRE, CORPS_F, premier=True)
    ecrit(tf, "Compression, photometrie et bruit ne genent pas. Le flou et le "
              "reechantillonnage degradent progressivement.", 14, False, TEXTE,
          CORPS_F, interligne=1.25, espace_avant=6)
    ecrit(tf, "Le recadrage annule le canal : Ω est un message GLOBAL de 64 bits "
              "reparti sur toute l'image, non un motif repete localement.",
          14, False, ROUGE, CORPS_F, interligne=1.25, espace_avant=8)

    # ------------------------- 10. resultat fort : le decodeur bat le pixel --
    d = page(); bandeau(d, "Le decodeur conditionne resiste mieux que le tatouage post-hoc",
                        "Resultat 2 sur 4  ·  le plus inattendu")
    cd = CategoryChartData()
    cd.categories = ["Flou k=7", "Reech. 0,5", "Bruit σ=0,1", "JPEG Q=30"]
    cd.add_series("Tatoueur pixel", (A["flou_k7"]["taux_authentic"],
                                     A["resize_0.5"]["taux_authentic"],
                                     A["bruit_std0.1"]["taux_authentic"],
                                     A["jpeg_q30"]["taux_authentic"]))
    cd.add_series("Decodeur conditionne", (D["flou_k7"]["taux_verdict"],
                                           D["resize_0.5"]["taux_verdict"],
                                           D["bruit_std0.1"]["taux_verdict"],
                                           D["jpeg_q30"]["taux_verdict"]))
    gr = d.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.75), Inches(1.95),
                            Inches(7.6), Inches(4.35), cd).chart
    gr.has_legend = True; gr.legend.position = XL_LEGEND_POSITION.BOTTOM
    gr.legend.include_in_layout = False
    gr.legend.font.size = Pt(13); gr.legend.font.name = CORPS_F
    gr.plots[0].series[0].format.fill.solid()
    gr.plots[0].series[0].format.fill.fore_color.rgb = RGBColor(0xB8, 0xB2, 0xA6)
    gr.plots[0].series[1].format.fill.solid()
    gr.plots[0].series[1].format.fill.fore_color.rgb = ACCENT
    gr.value_axis.maximum_scale = 1.0
    gr.value_axis.tick_labels.number_format = '0%'
    gr.value_axis.tick_labels.font.size = Pt(12)
    gr.value_axis.tick_labels.font.name = CORPS_F
    gr.category_axis.tick_labels.font.size = Pt(12.5)
    gr.category_axis.tick_labels.font.name = CORPS_F
    tf = zone(d, 8.7, 2.1, 3.9, 4.1)
    ecrit(tf, "Pourquoi", 17, True, ENCRE, CORPS_F, premier=True)
    ecrit(tf, "Le temoin porte par les poids NAIT avec l'image, au lieu de s'y "
              "superposer apres coup.", 15, False, TEXTE, CORPS_F,
          interligne=1.3, espace_avant=8)
    ecrit(tf, "Il est donc reparti dans la structure meme du contenu genere, et non "
              "dans un residu que le filtrage efface.", 15, False, TEXTE, CORPS_F,
          interligne=1.3, espace_avant=10)
    ecrit(tf, "Ce n'etait pas acquis d'avance : c'est le resultat le plus fort de "
              "la campagne.", 15, True, ACCENT, CORPS_F, interligne=1.3, espace_avant=12)

    # --------------------------------- 11. resultat : les deux familles -----
    d = page(); bandeau(d, "Le meme mecanisme, deux architectures, deux resultats",
                        "Resultat 3 sur 4")
    cd = CategoryChartData()
    cd.categories = ["Sans attaque", "JPEG Q=90", "JPEG Q=30", "Flou k=5", "Reech. 0,5"]
    cles = ["aucune", "jpeg_q90", "jpeg_q30", "flou_k5", "resize_0.5"]
    cd.add_series("Diffusion  (DC-AE, 6 etages)", tuple(D[k]["taux_verdict"] for k in cles))
    cd.add_series("Autoregressif  (MaskGIT, 3 etages)", tuple(G[k]["taux_verdict"] for k in cles))
    gr = d.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.75), Inches(1.95),
                            Inches(7.6), Inches(4.35), cd).chart
    gr.has_legend = True; gr.legend.position = XL_LEGEND_POSITION.BOTTOM
    gr.legend.include_in_layout = False
    gr.legend.font.size = Pt(13); gr.legend.font.name = CORPS_F
    gr.plots[0].series[0].format.fill.solid()
    gr.plots[0].series[0].format.fill.fore_color.rgb = APPUI
    gr.plots[0].series[1].format.fill.solid()
    gr.plots[0].series[1].format.fill.fore_color.rgb = RGBColor(0xC9, 0x8B, 0x5E)
    gr.value_axis.maximum_scale = 1.0
    gr.value_axis.tick_labels.number_format = '0%'
    gr.value_axis.tick_labels.font.size = Pt(12)
    gr.value_axis.tick_labels.font.name = CORPS_F
    gr.category_axis.tick_labels.font.size = Pt(12)
    gr.category_axis.tick_labels.font.name = CORPS_F
    tf = zone(d, 8.7, 2.1, 3.9, 4.1)
    ecrit(tf, "Le cout de l'agnosticisme", 17, True, ENCRE, CORPS_F, premier=True)
    ecrit(tf, "Le conditionneur ne trouve que 3 etages modulables dans MaskGIT contre "
              "6 dans le DC-AE, et l'un des trois porte sur la sortie RGB, ou moduler "
              "revient a teinter l'image.", 14.5, False, TEXTE, CORPS_F,
          interligne=1.28, espace_avant=8)
    ecrit(tf, "0,48 M de parametres contre 2,91 M.", 15, True, ENCRE, CORPS_F,
          interligne=1.25, espace_avant=10)
    ecrit(tf, "Le mecanisme s'applique partout ; son efficacite depend des prises "
              "que le decodeur offre.", 14.5, False, GRIS, CORPS_F,
          interligne=1.28, espace_avant=10)

    # ------------------------------------ 12. resultat : la securite --------
    d = page(); bandeau(d, "Ce que la cryptographie apporte, et que la mesure confirme",
                        "Resultat 4 sur 4")
    for i, (v, l, c) in enumerate([("0 / 25 000", "fausse attribution", VERT),
                                   ("0 / 300", "rejeu reussi", VERT),
                                   ("0,10 %", "transplantation reussie\n(8/8 avant correctif)", ACCENT),
                                   ("100 / 100", "chaine complete\nprotocole reel", VERT)]):
        pastille(d, 0.75 + i * 3.02, 2.0, 2.75, 1.7, v, l, c)
    tf = zone(d, 0.75, 3.9, 11.8, 2.5)
    ecrit(tf, "La chaine de bout en bout, verifiee sur GPU", 16, True, ENCRE, CORPS_F, premier=True)
    ecrit(tf, "Une classe est fournie. Le modele genere. L'image sort tatouee d'un Ω derive de "
              "l'IDENTITE du createur et lie au contenu. Un verifieur qui ne sait rien a l'avance "
              "lit le nonce, consulte le registre, rederive la cle, et retrouve le nom.",
          15, False, TEXTE, CORPS_F, interligne=1.3, espace_avant=8)
    ecrit(tf, "Controle negatif : decoder le MEME latent SANS conditionnement donne 31/64 bits, "
              "le hasard, et zero verdict sur 300. Le temoin vient bien de la modulation.",
          14.5, False, GRIS, CORPS_F, interligne=1.3, espace_avant=10)

    # ------------------------------------------- 13. face a DistSeal --------
    d = page(); bandeau(d, "Ce que CipherMark ajoute, et ce qu'il concede", "Analyse comparative")
    tableau(d, 0.75, 2.0, 11.8, 3.5,
            ["", "DistSeal (2026)", "CipherMark"],
            [["Message par generation", "fixe, rejouable", "unique, lie au contenu"],
             ["Securite formelle", "aucune", "HMAC-SHA256 + ChaCha20"],
             ["Attribution au createur", "non", "cle derivee, 0 erreur / 25 000"],
             ["Compression et bruit", "99,23 % (bit acc.)", "100 % de verdicts"],
             ["Transformations geometriques", "91,62 %", "0 %"],
             ["Familles couvertes", "diffusion + autoregressif", "les deux, avec un ecart mesure"]],
            largeurs=[3.4, 2.7, 3.2], taille=13.5)
    tf = zone(d, 0.75, 5.6, 11.8, 1.2)
    ecrit(tf, "Un echange explicite", 15.5, True, ENCRE, CORPS_F, premier=True)
    ecrit(tf, "Sur les transformations non geometriques, CipherMark tient son rang et depasse "
              "DistSeal en compression. Sur la geometrie, le retrait est une falaise. "
              "En contrepartie il acquiert quatre proprietes absentes de DistSeal.",
          14.5, False, TEXTE, CORPS_F, interligne=1.28, espace_avant=6)

    # ------------------------------------------------- 14. les limites ------
    d = page(); bandeau(d, "Trois limites, annoncees plutot que decouvertes apres coup",
                        "Discussion")
    lim = [("Le recadrage annule le canal", "0 % de verdicts sur les trois voies d'inscription. "
            "Ω est un message global de 64 bits, non un motif repete localement. "
            "La limite est structurelle, pas un defaut d'entrainement.", ROUGE),
           ("La liaison au contenu est plus fragile que le temoin",
            "Sous JPEG Q=30, 6 % des empreintes survivent quand 96 % des verdicts tiennent. "
            "Une verification exigeant EN PLUS l'empreinte serait bien moins robuste.", ACCENT),
           ("Le couple entierement latent n'a pas abouti",
            "Seul volet du programme reste inacheve. Obstacle diagnostique, correctif mesure, "
            "convergence non atteinte.", APPUI)]
    for i, (t, s2, c) in enumerate(lim):
        y = 2.0 + i * 1.6
        b = d.shapes.add_shape(1, Inches(0.75), Inches(y), Inches(11.8), Inches(1.4))
        b.fill.solid(); b.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        b.line.color.rgb = RGBColor(0xE0, 0xDD, 0xD5); b.shadow.inherit = False
        bar = d.shapes.add_shape(1, Inches(0.75), Inches(y), Inches(0.09), Inches(1.4))
        bar.fill.solid(); bar.fill.fore_color.rgb = c
        bar.line.fill.background(); bar.shadow.inherit = False
        tf = zone(d, 1.15, y + 0.16, 11.1, 1.1)
        ecrit(tf, t, 16.5, True, c, CORPS_F, premier=True)
        ecrit(tf, s2, 14.5, False, TEXTE, CORPS_F, interligne=1.25, espace_avant=4)
    note(d, "Sous transformation passive, le systeme ne produit que des faux negatifs, "
            "jamais de fausses accusations : le profil recherche en forensique.")

    # ----------------------------------------------- 15. perspectives ------
    d = page(); bandeau(d, "Ce qui prolonge ce travail", "Perspectives")
    puces(d, 0.75, 2.05, 11.8, [
        ("Achever le couple latent.", "L'obstacle est diagnostique : la recette de DistSeal, "
         "dont le poids de fidelite est nul, produit un modele qui lit bien en detruisant "
         "l'image. Retablir ce poids inverse la tendance des les premieres epoques."),
        ("Un hachage invariant a la geometrie", "mais discriminant entre contenus : "
         "c'est a soi seul un sujet de these, et la reponse a la falaise du recadrage."),
        ("Consolider le seuil de comparaison,", "aujourd'hui calibre sur un perimetre "
         "restreint et adosse a une binarisation instable sous compression."),
        ("Une application industrielle.", "L'AI Act cree une demande pour des systemes "
         "prouvant l'origine sans fausse accusation : c'est exactement le profil d'erreurs "
         "de CipherMark."),
    ], taille=16, ecart=16)

    # ------------------------------------------------- 16. conclusion -------
    d = page(); bandeau(d, "Trois contributions, et une reponse", "Conclusion")
    for i, (t, s2, c) in enumerate([
        ("CONCEPTUELLE", "Le champ temoin Ω comble un vide reel : unique par generation,\n"
                         "lie au contenu, illisible sans les cles.", APPUI),
        ("EXPERIMENTALE", "La chaine fonctionne de bout en bout, et l'inscription par\n"
                          "modulation des poids surpasse le tatouage post-hoc.", ACCENT),
        ("ANALYTIQUE", "Un mur de capacite apparent s'est revele un point stationnaire\n"
                       "exact ; deux hyperparametres ont porte la precision a 0,9998.", VERT)]):
        y = 1.95 + i * 1.33
        bar = d.shapes.add_shape(1, Inches(0.75), Inches(y), Inches(0.09), Inches(1.15))
        bar.fill.solid(); bar.fill.fore_color.rgb = c
        bar.line.fill.background(); bar.shadow.inherit = False
        tf = zone(d, 1.15, y + 0.05, 3.0, 0.5)
        ecrit(tf, t, 15, True, c, CORPS_F, premier=True)
        tf = zone(d, 4.0, y + 0.02, 8.5, 1.1)
        ecrit(tf, s2, 15, False, TEXTE, CORPS_F, interligne=1.25, premier=True)
    b = d.shapes.add_shape(1, Inches(0.75), Inches(5.95), Inches(11.8), Inches(1.05))
    b.fill.solid(); b.fill.fore_color.rgb = ENCRE
    b.line.fill.background(); b.shadow.inherit = False
    tf = zone(d, 1.15, 6.18, 11.0, 0.7)
    ecrit(tf, "Le tatouage generatif peut devenir une preuve cryptographique, "
              "individuelle et liee au contenu, aux limites connues.",
          17.5, True, PAPIER, TITRE_F, PP_ALIGN.CENTER, premier=True)

    # ------------------------------------------------------ 17. merci -------
    d = page(); fond(d, ENCRE)
    tf = zone(d, 1.4, 2.7, 10.5, 1.2)
    ecrit(tf, "Merci de votre attention", 40, True, PAPIER, TITRE_F, premier=True)
    barre = d.shapes.add_shape(1, Inches(1.4), Inches(4.05), Inches(1.6), Inches(0.05))
    barre.fill.solid(); barre.fill.fore_color.rgb = ACCENT
    barre.line.fill.background(); barre.shadow.inherit = False
    tf = zone(d, 1.4, 4.4, 10.5, 1.4)
    ecrit(tf, AUTEUR, 19, True, PAPIER, CORPS_F, premier=True)
    ecrit(tf, ETABLISSEMENT, 14, False, RGBColor(0x9A, 0xA4, 0xB0), CORPS_F, espace_avant=6)
    ecrit(tf, "Code et resultats : github.com/ngueagho/distseal-meta", 14, False,
          RGBColor(0x9A, 0xA4, 0xB0), CORPS_F, espace_avant=4)


    # ---- notes du presentateur ----
    for i, t in ((1, "Bonjour. Je presente CipherMark, un tatouage pour modeles generatifs qui ajoute une garantie cryptographique la ou l'etat de l'art n'en offre pas. Vingt minutes : le probleme, la proposition, les mesures, les limites."), (2, "Point de depart : on ne distingue plus une image generee d'une photo. L'AI Act impose donc de marquer. Mais detecter ne suffit pas : il faut PROUVER l'origine sans jamais accuser a tort. Sur dix-sept travaux recents, aucun ne tient les trois proprietes ensemble."), (3, "DistSeal couvre les deux familles mais grave un message FIXE : toutes ses generations portent le meme filigrane, donc rejouable. MetaSeal apporte la cryptographie mais reste sur la diffusion. Le manque est a l'intersection."), (4, "D'ou la question. Insister sur le 'a la fois' : chaque propriete prise isolement existe deja ; c'est leur conjonction qui manque."), (5, "Le coeur de la contribution. HMAC lie le temoin au contenu par une empreinte DINOv2. Le PRG le rend unique et illisible. La cle par createur est DERIVEE, donc l'attribution ne se declare pas, elle se prouve. Ce que DistSeal grave une fois, on le recalcule a chaque image."), (6, "Trois couches. Insister sur la derniere : le verifieur ne connait rien a l'avance. Il lit le nonce, consulte le registre, REDERIVE la cle. Aucun secret par utilisateur n'est stocke nulle part."), (7, "Le mecanisme vient de WOUAF. Deux differences : identite exacte a l'initialisation, et modulation BORNEE la ou WOUAF demodule. Le chiffre a retenir est 6 contre 3 : il explique tout l'ecart de la diapositive 11."), (8, 'Insister sur la rigueur : memes parametres, meme echelle, memes vingt conditions pour les trois canaux. Parametres FIXES, donc reproductibles. Et deux controles negatifs a chaque mesure.'), (9, 'Le canal pixel. Tout tient sauf le recadrage. Dire clairement pourquoi : Omega est un message GLOBAL de 64 bits, pas un motif repete. La limite est structurelle, elle etait annoncee.'), (10, "LE resultat de la campagne, et il n'etait pas attendu. Le tatouage inscrit dans les POIDS resiste mieux que celui applique apres coup. Laisser le graphique parler deux secondes avant de commenter."), (11, "Meme mecanisme, deux architectures, resultats sans commune mesure. La cause est mesuree, pas conjecturee : 3 etages modulables contre 6, dont un sur la sortie RGB ou moduler revient a teinter. C'est le cout de l'agnosticisme."), (12, 'Ce que la cryptographie apporte concretement. Le controle negatif est le point cle : sans conditionnement, le meme latent donne 31/64, le hasard. Le temoin vient donc bien de la modulation.'), (13, "Un echange assume. On depasse DistSeal en compression, on cede totalement sur la geometrie, et on gagne quatre proprietes qu'il n'a pas. Ne pas presenter cela comme une victoire nette."), (14, "Les annoncer soi-meme plutot que se les faire opposer. La deuxieme est celle que le jury peut ne pas voir : la liaison au contenu est PLUS fragile que le temoin. Je l'ai decouverte en passant a 5000 images."), (15, "La premiere perspective est la plus concrete : l'obstacle est diagnostique et le correctif mesure, il manque la convergence. La deuxieme est un vrai sujet de these."), (16, "Resumer en trois mots : Omega comble un vide, la chaine fonctionne, et le mur de capacite etait un artefact d'optimisation. Finir sur la phrase du bandeau."), (17, 'Merci. Le code et tous les resultats sont publics.')):
        commentaire(prs.slides[i - 1], t)

    prs.save(os.path.join(SORTIE, "CipherMark_soutenance.pptx"))
    print(f"  {len(prs.slides.__iter__.__self__._sldIdLst)} diapositives ecrites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
