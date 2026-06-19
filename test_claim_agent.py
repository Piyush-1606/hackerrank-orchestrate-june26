from __future__ import annotations

from code.agents.claim_agent import ClaimAgent
from code.models.schemas import ClaimInput, IssueType, ObjectType, Severity

print("TEST STARTED")

def _run_agent(claim_text: str, object_type: ObjectType) -> object:
    claim = ClaimInput(
        claim_id="test_claim",
        user_id="user_001",
        image_paths=["images/test/img_1.jpg"],
        claim_text=claim_text,
        object_type=object_type,
        user_history=None,
        evidence_requirements=[],
        metadata={},
    )
    return ClaimAgent().run(claim)


def test_rear_bumper_dent_uses_all_customer_turns() -> None:
    result = _run_agent(
        "Customer: I found a dent now on the rear bumper. | "
        "Agent: Should we review the back side of the vehicle? | "
        "Customer: Yes, please check that photo.",
        ObjectType.CAR,
    )

    assert IssueType.DENT in result.claimed_issue_types
    assert result.affected_area == "rear_bumper"
    assert result.claimed_severity == Severity.MEDIUM


def test_windshield_crack_spreading_from_earlier_turn() -> None:
    result = _run_agent(
        "Customer: The windshield has a crack spreading across it. | "
        "Support: Is it only the glass? | "
        "Customer: Yes, that is the issue.",
        ObjectType.CAR,
    )

    assert IssueType.CRACK in result.claimed_issue_types
    assert result.affected_area == "windshield"
    assert result.claimed_severity == Severity.MEDIUM


def test_side_mirror_damage_generic_damage_keyword() -> None:
    result = _run_agent(
        "Customer: My side mirror is damaged after parking. | "
        "Agent: Anything else? | "
        "Customer: No, only that part.",
        ObjectType.CAR,
    )

    assert result.claimed_issue_types[0] == IssueType.UNKNOWN
    assert result.affected_area == "side_mirror"
    assert result.claimed_severity == Severity.MEDIUM


def test_side_mirror_not_sitting_correctly_is_broken_part() -> None:
    result = _run_agent(
        "Customer: The side mirror got damaged. It is not sitting the way it should anymore. | "
        "Support: Was there damage to the door too? | "
        "Customer: No, just the mirror.",
        ObjectType.CAR,
    )

    assert result.claimed_issue_types[0] == IssueType.BROKEN_PART
    assert result.affected_area == "side_mirror"


def test_shattered_laptop_screen_normalizes_to_crack() -> None:
    result = _run_agent(
        "Customer: My laptop screen looks shattered to me. | "
        "Support: Is the hinge affected? | "
        "Customer: No, only the shattered screen.",
        ObjectType.LAPTOP,
    )

    assert result.claimed_issue_types[0] == IssueType.CRACK
    assert result.affected_area == "screen"


def test_ambiguous_trackpad_physical_damage_is_none() -> None:
    result = _run_agent(
        "Customer: The laptop trackpad has stopped working properly. | "
        "Support: Are you reporting internal function or physical damage? | "
        "Customer: Physical damage around the trackpad area.",
        ObjectType.LAPTOP,
    )

    assert result.claimed_issue_types[0] == IssueType.NONE
    assert result.affected_area == "trackpad"


def test_package_corner_crushed() -> None:
    result = _run_agent(
        "Cliente: La esquina del paquete llegó crushed. | "
        "Soporte: ¿El contenido está dañado? | "
        "Cliente: Solo package corner crushed, please review.",
        ObjectType.PACKAGE,
    )

    assert IssueType.CRUSHED_PACKAGING in result.claimed_issue_types
    assert result.affected_area == "package_corner"
    assert result.claimed_severity == Severity.MEDIUM

if __name__ == "__main__":
    print("Running ClaimAgent tests...")

    test_rear_bumper_dent_uses_all_customer_turns()
    print("✓ rear bumper dent")

    test_windshield_crack_spreading_from_earlier_turn()
    print("✓ windshield crack")

    test_side_mirror_damage_generic_damage_keyword()
    print("✓ side mirror")

    test_package_corner_crushed()
    print("✓ package corner")

    print("All tests passed!")
