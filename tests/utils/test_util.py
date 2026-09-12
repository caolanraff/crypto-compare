from src.utils.util import convert_to_snake_case


def test_convert_to_snake_case() -> None:
    assert convert_to_snake_case("Financial Data") == "financial_data"
    assert convert_to_snake_case("already_snake") == "already_snake"
