"""Normalización de nombres de selección (mundial.nombres)."""
from mundial import nombres


def test_alias_directos():
    assert nombres.normaliza("USA") == "United States"
    assert nombres.normaliza("Korea Republic") == "South Korea"
    assert nombres.normaliza("Cabo Verde") == "Cape Verde"
    assert nombres.normaliza("Türkiye") == "Turkey"


def test_grafias_del_dataset_colisionadas():
    # el alias corrige incluso grafías que el propio dataset trae inconsistentes
    assert nombres.normaliza("Congo DR") == "DR Congo"
    assert nombres.normaliza("DR Congo") == "DR Congo"
    # República del Congo es OTRO país y no debe unirse
    assert nombres.normaliza("Congo") == "Congo"


def test_normalizacion_por_tokens():
    assert nombres.normaliza("The Gambia") == "Gambia"
    assert nombres.normaliza("Bosnia-Herzegovina") == "Bosnia and Herzegovina"
    assert nombres.normaliza("St. Vincent and the Grenadines") == \
        "Saint Vincent and the Grenadines"


def test_desconocido_pasa_intacto():
    assert nombres.normaliza("Atlántida FC") == "Atlántida FC"
    assert nombres.normaliza("") == ""


def test_resolver_con_canonicos_propios():
    r = nombres.Resolver({"Alpha United", "Beta City"})
    assert r.resolve("alpha united") == "Alpha United"
    assert r.resolve("Beta  City") == "Beta City"
    assert r.resolve("Gamma Town") == "Gamma Town"  # sin match -> intacto
