"""Smoke test de la app completa (marcado slow: entrena el pipeline entero).

Es el test más valioso del repo: la mayoría de los fallos históricos fueron
de render en Streamlit (APIs deprecadas, tipos de columna, keys duplicadas),
no de lógica. Corre en su propio job de CI.
"""
import pytest


@pytest.mark.slow
def test_app_principal_renderiza_sin_excepciones():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=900).run()
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.slow
def test_editor_mundial_renderiza_sin_excepciones():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("pages/1_⚽_Editor_Mundial.py",
                           default_timeout=900).run()
    assert not at.exception, [str(e) for e in at.exception]
