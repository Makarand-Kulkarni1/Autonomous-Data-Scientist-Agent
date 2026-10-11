from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """One task in the plan. The code agent will carry this out."""
    id: int = Field(description="Step number, starting at 1")
    title: str = Field(description="Short name, e.g. 'Handle missing values'")
    task: str = Field(description="A precise, self-contained instruction for a coder to follow")


class Plan(BaseModel):
    """The full plan the planner agent produces."""
    target_column: str = Field(description="The column being predicted")
    problem_type: str = Field(description="Either 'classification' or 'regression'")
    steps: list[PlanStep]
