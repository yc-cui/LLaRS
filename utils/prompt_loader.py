"""Load prompt JSON files."""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional


def load_prompts(prompt_file: Optional[str]) -> Dict[str, list]:
    """
    Load prompts JSON:
    - If prompt_file is None, use <project_root>/prompts/all_prompts.json
    - Otherwise use the given path
    """
    if prompt_file is None:
        project_root = Path(__file__).parent.parent
        path = project_root / "prompts" / "all_prompts.json"
    else:
        path = Path(prompt_file)

    with open(path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    return prompts


def select_prompt_for_deg_type(
    prompts: Dict[str, List[str]],
    deg_type: str,
    is_test: bool,
    rng: random.Random,
) -> str:
    """Pick one prompt for deg_type: first entry at test, random at train."""
    lst = prompts[deg_type]
    assert lst, f"prompt list for deg_type '{deg_type}' is empty"
    if is_test:
        return lst[0]
    return rng.choice(lst)


