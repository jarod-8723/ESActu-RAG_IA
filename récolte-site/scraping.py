# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 20:28:27 2026

@author: Jarod
"""

import os
import re
import time
import requests
import html2text
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from datetime import date
from pathlib import Path
import copy
import pymupdf4llm
import tempfile
import sys
import io


class SiteScraper:

    SECTION_MAP = {
        # --- Formation ---
        "la-formation":             "Formation",
        "presentation-du-cursus":   "Formation",
        "la-premiere-annee":        "Formation",
        "la-seconde-annee":         "Formation",
        "master-esa-2eme-annee-semestre-4-voie-recherche": "Formation",
        "evaluation-de-la-formation": "Formation",

        # --- Candidature ---
        "candidature-au-master-esa": "Candidature",

        # --- Equipe ---
        "equipe-pedagogique":       "Equipe",

        # --- Alumni ---
        "nos-etudiants":            "Alumni",
        "promotion":                "Alumni",
        "les-promotions-passees":   "Alumni",

        # --- Insertion ---
        "insertion-professionnelle": "Insertion",
        "stages":                   "Insertion",
        "offres-de-stages":         "Insertion",
        "la-convention-de-stage":   "Insertion",
        "rapport-de-stage-et-soutenance": "Insertion",
        "emplois":                  "Insertion",
        "chiffres-insertion":       "Insertion",
        "panorama-insertion-de-la-voie-recherche": "Insertion",
        "master-esa-15-ans-dinsertion": "Insertion",
        "entreprises-partenaires":  "Insertion",

        # --- Ressources ---
        "ressources":               "Ressources",
        "actualites":               "Ressources",
        "partenariat-cacl-pret-etudiant": "Ressources",
        "liens-utiles":             "Ressources",
        "ressources-la-lettre":     "Ressources",

        # --- Contact ---
        "contact":                  "Contact",
        "plan-du-site":             "Contact",
        "mentions-legales":         "Contact",
        "politique-de-confidentialite": "Contact",
    }

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.visited: set[str] = set()
        self.all_pdfs: set[str] = set()

        self.base_dir = Path(__file__).parent.parent / "data"
        self.base_dir.mkdir(exist_ok=True)





    def _extract_year_range(self, text: str) -> tuple[int, int] | None:
        """Cherche un pattern XXXX-XXXX ou XXXX / XXXX dans un texte."""
        m = re.search(r"(20[0-2]\d)[^\d]*(20[0-2]\d)", text)
        if m:
            y1, y2 = int(m.group(1)), int(m.group(2))
            if 2004 <= y1 <= 2100 and y2 == y1 + 1:
                return y1, y2
        return None




    def _extract_year_range_from_soup(self, soup: BeautifulSoup) -> tuple[int, int] | None:
        """Cherche une fourchette d'années dans le contenu HTML de la page."""
        if not soup:
            return None
        # Cherche dans le titre <title> en priorité
        title_tag = soup.find("title")
        if title_tag:
            result = self._extract_year_range(title_tag.get_text())
            if result:
                return result
        # Puis dans les headings
        for tag in soup.find_all(["h1", "h2", "h3"]):
            result = self._extract_year_range(tag.get_text())
            if result:
                return result
        return None




    def _url_to_path(self, url: str, soup: BeautifulSoup | None = None,
                     page_title: str = "") -> Path:
        """Retourne le chemin de fichier .md pour une URL donnée."""
        segments = [s for s in urlparse(url).path.strip("/").split("/") if s]

        section = "Accueil"
        filename = "index"
        year_range: tuple[int, int] | None = None

        if segments:
            for seg in segments:
                if seg in self.SECTION_MAP:
                    section = self.SECTION_MAP[seg]
                    break
            filename = segments[-1] if segments else "index"

        
        if section == "Alumni":
            # 1. Cherche dans les segments de l'URL
            full_path = urlparse(url).path
            year_range = self._extract_year_range(full_path)

            # 2. Fallback : cherche dans le HTML (titre, headings)
            if not year_range and soup:
                year_range = self._extract_year_range_from_soup(soup)

        folder = self.base_dir / section

        if year_range:
            folder = folder / f"Promotion_{year_range[0]}-{year_range[1]}"
        elif section == "Alumni":
            folder = folder / "Promotion_Inconnue"

        folder.mkdir(parents=True, exist_ok=True)

        # Sanitize le nom de fichier (évite les slugs trop longs)
        safe_filename = re.sub(r"[^\w\-]", "_", filename)[:80]
        return folder / f"{safe_filename}.md"
    
    
    

    def fetch_page(self, url: str) -> BeautifulSoup | None:
        """Télécharge et parse une page HTML."""
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser")
        except requests.RequestException:
            print(f"  ✗ Erreur téléchargement : {url}")
            return None




    def _get_page_title(self, soup: BeautifulSoup) -> str:
        """Extrait le titre propre de la page."""
        title_tag = soup.find("title")
        if title_tag:
            # Retire le suffixe " - Master ESA" courant
            title = title_tag.get_text(strip=True)
            title = re.sub(r"\s*[-|]\s*Master ESA.*$", "", title, flags=re.IGNORECASE)
            return title.strip()
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        return ""




    def extract_main_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Nettoie le HTML des éléments inutiles pour l'ingestion."""
        # Supprime les balises structurelles de navigation/formulaires
        for tag in soup.find_all(["header", "footer", "nav", "form",
                                   "script", "style", "aside"]):
            tag.decompose()

        # Supprime les divs de navigation/social par classe
        blocklist_classes = re.compile(
            r"login|sidebar|widget|cookie|popup|navbar|menu|nav|button|social|share|comment",
            re.I
        )
        for tag in soup.find_all("div", class_=blocklist_classes):
            tag.decompose()

        # Supprime les sections par id
        for tag in soup.find_all(["div", "section"], id=re.compile(
            r"cookie|login|signin|modal|popup|sidebar", re.I
        )):
            tag.decompose()

        # FIX : supprime les listes <ul>/<li> de navigation qui restent
        # Le site duplique son menu 3 fois dans le HTML — on cible les <ul> avec
        # beaucoup de <li> imbriqués qui sont typiquement du nav
        for ul in soup.find_all("ul"):
            links = ul.find_all("a")
            # Une liste purement navigationnelle a beaucoup de liens, peu de texte libre
            if len(links) > 5:
                texts = [a.get_text(strip=True) for a in links]
                nav_keywords = {"La Formation", "Nos Étudiants", "Insertion professionnelle",
                                 "Ressources", "Contact", "Espace étudiant"}
                if any(kw in texts for kw in nav_keywords):
                    ul.decompose()

        return soup






    def html_to_markdown(self, soup: BeautifulSoup, source_url: str,
                         page_title: str = "", section: str = "",
                         year_range: tuple | None = None) -> str:
        """Convertit une page HTML en markdown enrichi pour l'ingestion LLM."""
        soup_copy = copy.deepcopy(soup)
        soup_copy = self.extract_main_content(soup_copy)

        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.inline_links = False
        converter.ignore_images = True
        converter.body_width = 0  # Pas de retour à la ligne automatique

        md = converter.handle(str(soup_copy))
        md = self.clean_markdown(md)

        
        promotion_str = (f"Promotion_{year_range[0]}-{year_range[1]}"
                         if year_range else "")
        frontmatter_parts = [
            "---",
            f"url: {source_url}",
            f"scraped_at: {date.today()}",
        ]
        if page_title:
            frontmatter_parts.append(f'title: "{page_title}"')
        if section:
            frontmatter_parts.append(f"section: {section}")
        if promotion_str:
            frontmatter_parts.append(f"promotion: {promotion_str}")
        frontmatter_parts.append("---\n")

        frontmatter = "\n".join(frontmatter_parts)
        return frontmatter + md






    def clean_markdown(self, md: str) -> str:
        """Nettoie le markdown des bruits inutiles pour l'ingestion."""

        NOISE_PATTERNS = [
            # Navigation / interface
            r"Passer au contenu\s*\n",
            # FIX : le tag XML affiché en haut de chaque page du site
            r"\[xml version.*?\]\([^)]*\)\s*",
            # Liens images
            r"\[!\[.*?\]\(.*?\)\]\(.*?\)",
            # Séparateurs vides
            r"\*\s*\*\s*\*",
            # Formulaires de login
            r"(?i)(connexion|se souvenir de moi|mot de passe oublié.*?\n|email\s*\n|mot de passe\s*\n)",
            # Liens d'action inutiles
            r"(?i)(\[s'inscrire.*?\]\(.*?\))",
            r"(?i)(\[suivez.*?\]\(.*?\))",
            r"(?i)(\[partager.*?\]\(.*?\))",
            r"(?i)(\[imprimer.*?\]\(.*?\))",
            # Lignes vides multiples
            r"\n{3,}",
            # Lignes ne contenant qu'un lien (souvent navigation)
            r"^\[.*?\]\(.*?\)\s*$",
        ]

        for pattern in NOISE_PATTERNS:
            md = re.sub(pattern, "", md, flags=re.MULTILINE | re.IGNORECASE | re.DOTALL)

        lines = md.split('\n')
        filtered_lines = []
        skip = False

        for line in lines:
            lower_line = line.lower()
            if re.match(r"^#+\s", line):
                if any(kw in lower_line for kw in
                       ["connexion", "newsletter", "cookies", "commentaires", "social"]):
                    skip = True
                    continue
                else:
                    skip = False
            if not skip:
                filtered_lines.append(line)

        md = '\n'.join(filtered_lines)
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md.strip()
    
    
    
    

    def _save_pdf(self, pdf_url: str, parent_url: str,
                  soup: BeautifulSoup | None = None):
        """Convertit un PDF en fichier markdown et le sauvegarde."""
        try:
            r = requests.get(pdf_url, timeout=15)
            r.raise_for_status()

            # FIX : on passe aussi le soup parent pour bien classer les PDF Alumni
            folder = self._url_to_path(parent_url, soup=soup).parent
            folder.mkdir(parents=True, exist_ok=True)
            filename = re.sub(r"[^\w\-]", "_", pdf_url.split("/")[-1].replace(".pdf", ""))[:80]

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(r.content)
                tmp_path = tmp.name

            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                md_text = pymupdf4llm.to_markdown(tmp_path)
            finally:
                sys.stderr = old_stderr
                os.remove(tmp_path)

            md_text = md_text.encode("utf-8", errors="ignore").decode("utf-8")
            frontmatter = (f"---\nurl: {pdf_url}\nscraped_at: {date.today()}"
                           f"\ntype: pdf\nsource_page: {parent_url}\n---\n\n")
            filepath = folder / f"{filename}.md"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(frontmatter + md_text)

            print(f"  ✓ PDF → Markdown : {filepath}")

        except Exception as e:
            print(f"  ✗ Erreur PDF : {pdf_url} ({e})")






    def save_markdown(self, url: str, md: str,
                      soup: BeautifulSoup | None = None,
                      page_title: str = ""):
        """Sauvegarde un fichier markdown si le contenu est suffisant."""
        word_count = len(md.split())
        if word_count < 50:
            print(f"  ⚠️  Ignoré ({word_count} mots) : {url}")
            return

        filepath = self._url_to_path(url, soup=soup)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md)






    def _get_section_for_url(self, url: str) -> str:
        """Retourne la section associée à une URL."""
        segments = [s for s in urlparse(url).path.strip("/").split("/") if s]
        for seg in segments:
            if seg in self.SECTION_MAP:
                return self.SECTION_MAP[seg]
        return "Accueil"






    def crawl(self):
        """Fonction principale de scraping."""
        to_visit = [self.base_url]

        while to_visit:
            current_url = to_visit.pop(0)
            if current_url in self.visited:
                continue
            self.visited.add(current_url)

            print(f"[SCRAPING] {current_url}")

            soup = self.fetch_page(current_url)
            if not soup:
                continue

            # Récupère les métadonnées utiles pour le frontmatter
            page_title = self._get_page_title(soup)
            section = self._get_section_for_url(current_url)
            year_range = None
            if section == "Alumni":
                year_range = self._extract_year_range(urlparse(current_url).path)
                if not year_range:
                    year_range = self._extract_year_range_from_soup(soup)

            md = self.html_to_markdown(
                soup,
                source_url=current_url,
                page_title=page_title,
                section=section,
                year_range=year_range,
            )
            self.save_markdown(current_url, md, soup=soup, page_title=page_title)

            # Découverte des liens
            for a in soup.find_all("a", href=True):
                url = urljoin(current_url, a["href"]).rstrip("/")
                if (
                    url not in self.visited
                    and urlparse(url).netloc == urlparse(self.base_url).netloc
                    and not url.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg"))
                    and "#" not in url
                    and "wp-login" not in url
                    and "wp-admin" not in url
                    and "member-login" not in url
                ):
                    if url.endswith(".pdf"):
                        if url not in self.all_pdfs:
                            self._save_pdf(url, parent_url=current_url, soup=soup)
                            self.all_pdfs.add(url)
                            self.visited.add(url)
                    else:
                        to_visit.append(url)

            time.sleep(1)

        print(f"\n[DONE] {len(self.visited)} pages scrapées | {len(self.all_pdfs)} PDFs")


if __name__ == "__main__":
    scraper = SiteScraper("https://www.master-esa.fr")
    scraper.crawl()