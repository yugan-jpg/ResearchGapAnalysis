"""Conservative deterministic checks; uncertain results require human review."""

import re
import unicodedata
from decimal import Decimal, InvalidOperation


def is_factually_rejected(record: dict) -> bool:
    """A rejected or explicitly unsupported claim is not a duplicate target."""
    return (record.get("verification_status") == "rejected"
            or record.get("factual_verdict") == "rejected"
            or record.get("evidence_supported") is False)


def measurement_tokens(text: str) -> set[str]:
    """Extract numeric values without treating gene names or citations as data.

    Superscript digits are excluded rather than NFKC-normalized into adjacent
    measurements. Original text remains unchanged for evidence matching.
    """
    # Treat superscript exponents as scientific notation before stripping
    # superscript references; never turn a citation into part of a year.
    superscript = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
    text = re.sub(r"10([⁺⁻][⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: "1e" + m.group(1).translate(superscript), text)
    text = text.replace("−", "-").replace("﹣", "-")
    text = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾]+", " ", text)
    text = re.sub(r"\[(?:\s*\d+\s*[,–\-]?\s*)+\]", " ", text)
    text = re.sub(r"(?<=\d)(?=(?:mg|kg|ml|mm|cm|µg|μg|ng|hz)\b)", " ", text, flags=re.IGNORECASE)
    # A slash-delimited suffix belongs to the identifier (plasmepsin2/3),
    # but only when the token also contains a letter.
    text = re.sub(
        r"(?<!\w)(?=[A-Za-z])(?=[A-Za-z0-9/_-]*\d)[A-Za-z0-9]+(?:[/_-][A-Za-z0-9]+)*",
        " ", text,
    )
    values = set()
    pattern = r"(?<![\w.])[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?\s*%?(?![\w]|\.\d)"
    for match in re.findall(pattern, text):
        raw = match.replace(",", "").replace(" ", "")
        percent = raw.endswith("%")
        try:
            number = Decimal(raw[:-1] if percent else raw).normalize()
        except InvalidOperation:
            continue
        values.add(format(number, "f") + ("%" if percent else ""))
    return values


def inequality_tokens(text: str) -> set[tuple[str, str]]:
    """Keep statistical comparison direction; numbers alone are insufficient."""
    text = text.replace("≤", "<=").replace("≥", ">=").replace("−", "-")
    found = set()
    for operator, number in re.findall(r"(<=|>=|<|>)\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)", text):
        values = measurement_tokens(number)
        if values:
            found.add((operator, next(iter(values))))
    return found


def normalize_claim(text: str) -> str:
    """Normalize presentation while keeping scientific symbols and negation."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(" .;")


def source_total_warning(source: str) -> str | None:
    """Flag a possible count inconsistency, without assuming disjoint groups."""
    total = re.search(
        r"\b(?:total(?:\s+of)?|overall)\s*[:=]?\s*(\d[\d,]*)\s*(?:samples?|participants?|subjects?|patients?|cases?)\b",
        source, re.IGNORECASE,
    )
    if not total:
        return None
    remainder = source[total.end():total.end() + 350]
    remainder = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾]+|\[(?:\s*\d+\s*[,–\-]?\s*)+\]", " ", remainder)
    remainder = re.sub(r"(?<!\w)(?=[A-Za-z0-9/_-]*[A-Za-z])(?=[A-Za-z0-9/_-]*\d)[A-Za-z0-9]+(?:[/_-][A-Za-z0-9]+)*", " ", remainder)
    numbers = re.findall(r"(?<![\w.])\d+(?:,\d{3})*(?![\w%]|\.\d)", remainder)
    counts = [int(item.replace(",", "")) for item in numbers]
    if len(counts) < 2 or len(counts) > 8:
        return None
    reported = int(total.group(1).replace(",", ""))
    if sum(counts) != reported:
        return (
            f"Source reports a total of {reported}; nearby counts sum to {sum(counts)}. "
            "Review whether the counts describe overlapping or different groups."
        )
    return None
