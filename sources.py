"""Публикации, на которые ссылаются формулы расчётной модели."""

from __future__ import annotations


SOURCES = {
    "HACOT_1998": {
        "short": "Hacot, Dubowsky, Bidaud (1998)",
        "title": "Analysis and Simulation of a Rocker-Bogie Exploration Rover",
        "url": "https://www.slideshare.net/slideshow/analysis-and-simulation-of-a-rocker-bogie-exploration-rover/49411474",
        "used_for": (
            "обратная кинематика двух сторон; средний угол корпуса через "
            "дифференциал; квазистатические уравнения (8–15); ограничения "
            "Ni ≥ 0, Ti ≤ μNi и τi ≤ τmax"
        ),
    },
    "PAPADOPOULOS_2000": {
        "short": "Papadopoulos, Rey (2000)",
        "title": "The Force-Angle Measure of Tipover Stability Margin for Mobile Manipulators",
        "url": "https://www.tandfonline.com/doi/abs/10.1076/0042-3114%28200001%2933%3A1%3B1-5%3BFT029",
        "used_for": (
            "опорный многоугольник и силоугловой запас до опрокидывания "
            "на неровной поверхности"
        ),
    },
    "LAMON_2004": {
        "short": "Lamon, Krebs et al. (2004)",
        "title": "Wheel Torque Control for a Rough Terrain Rover",
        "url": "https://infoscience.epfl.ch/bitstreams/b4940e69-33a4-4379-aad8-c35ca12d363f/download",
        "used_for": (
            "квазистатическое распределение моментов мотор-колёс, "
            "ограничение проскальзывания и загрузка приводов"
        ),
    },
    "NASA_HARRINGTON_2004": {
        "short": "Harrington, Voorhees (NASA, 2004)",
        "title": "The Challenges of Designing the Rocker-Bogie Suspension for the Mars Exploration Rover",
        "url": "https://ntrs.nasa.gov/citations/20040084284",
        "used_for": "устройство и конструктивные свойства rocker-bogie",
    },
    "NASA_LINDEMANN_2005": {
        "short": "Lindemann, Voorhees (NASA/JPL, 2005)",
        "title": "Mars Exploration Rover Mobility Assembly Design, Test and Performance",
        "url": "https://ntrs.nasa.gov/citations/20060042787",
        "used_for": "проверенная архитектура шестиколёсной подвески и её испытания",
    },
}

