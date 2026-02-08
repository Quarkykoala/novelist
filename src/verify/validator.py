"""
Hypothesis Validator — Quality checks for generated hypotheses.

This module validates hypotheses for:
- Falsifiability (can it be proven wrong?)
- Mechanism completeness (is the causal chain complete?)
- Number sanity (are predictions within reasonable bounds?)
- Specificity (is it concrete enough?)
"""

import re
from dataclasses import dataclass

from src.contracts.schemas import (
    ExtractedClaim,
    GroundedHypothesis,
    Hypothesis,
)


@dataclass
class ValidationResult:
    """Result of hypothesis validation."""
    is_valid: bool
    issues: list[str]
    suggestions: list[str]
    score: float  # 0-1 quality score


class HypothesisValidator:
    """Validates hypotheses for quality and actionability."""

    # Words that indicate vagueness
    VAGUE_WORDS = {
        "may", "might", "could", "possibly", "potentially",
        "it is possible", "it is likely", "perhaps", "seems",
        "various", "some", "many", "several", "significant",
    }

    # Words that indicate mechanism-free claims
    MECHANISM_FREE = {
        "improves", "enhances", "affects", "influences",
        "can lead to", "results in", "causes",
    }

    def validate_grounded(
        self,
        hypothesis: GroundedHypothesis,
        baselines: list[ExtractedClaim] | None = None,
    ) -> ValidationResult:
        """Validate a grounded hypothesis."""
        issues = []
        suggestions = []

        # Check 1: Falsifiability
        if not self._check_falsifiability(hypothesis):
            issues.append("Missing or vague null result — hypothesis not falsifiable")
            suggestions.append("Specify what observation would reject this hypothesis")

        # Check 2: Mechanism completeness
        mech_issues = self._check_mechanism(hypothesis)
        if mech_issues:
            issues.extend(mech_issues)
            suggestions.append("Ensure each mechanism step is causally connected")

        # Check 3: Number sanity (if baselines provided)
        if baselines and hypothesis.prediction_bounds:
            num_issues = self._check_number_sanity(hypothesis, baselines)
            if num_issues:
                issues.extend(num_issues)
                suggestions.append("Bound predictions within 2x of existing SOTA")

        # Check 4: Specificity
        spec_issues = self._check_specificity(hypothesis.claim)
        if spec_issues:
            issues.extend(spec_issues)
            suggestions.append("Replace vague terms with specific compounds, genes, or methods")

        # Check 5: Experiment quality
        if not hypothesis.suggested_experiments:
            issues.append("No experiments suggested")
            suggestions.append("Add at least one concrete experiment with controls")

        # Calculate score
        max_issues = 6
        score = max(0.0, 1.0 - (len(issues) / max_issues))

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            suggestions=suggestions,
            score=score,
        )

    def validate_standard(self, hypothesis: Hypothesis) -> ValidationResult:
        """Validate a standard hypothesis."""
        issues = []
        suggestions = []

        # Check specificity of main claim
        spec_issues = self._check_specificity(hypothesis.hypothesis)
        if spec_issues:
            issues.extend(spec_issues)
            suggestions.append("Make the hypothesis statement more specific")

        # Check experimental design
        if not hypothesis.experimental_design or len(hypothesis.experimental_design) == 0:
            issues.append("No experimental design provided")
            suggestions.append("Add step-by-step experimental protocol")

        # Check rationale
        if len(hypothesis.rationale) < 50:
            issues.append("Rationale too brief")
            suggestions.append("Explain why this hypothesis could be true and why it matters")

        # Calculate score
        max_issues = 4
        score = max(0.0, 1.0 - (len(issues) / max_issues))

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            suggestions=suggestions,
            score=score,
        )

    def _check_falsifiability(self, hypothesis: GroundedHypothesis) -> bool:
        """Check if hypothesis has a clear null result."""
        if not hypothesis.null_result:
            return False

        null = hypothesis.null_result.lower()

        # Check for vague null results
        if any(vague in null for vague in ["no change", "nothing happens", "unclear"]):
            return False

        # Must specify a measurable condition
        has_measurement = any(
            word in null for word in
            ["observe", "measure", "detect", "see", "%", "increase", "decrease"]
        )

        return has_measurement

    def _check_mechanism(self, hypothesis: GroundedHypothesis) -> list[str]:
        """Check mechanism chain quality."""
        issues = []

        if not hypothesis.mechanism:
            issues.append("No mechanism chain provided")
            return issues

        if len(hypothesis.mechanism) < 2:
            issues.append("Mechanism chain too short — need at least 2 steps")

        # Check for disconnected steps
        for i in range(len(hypothesis.mechanism) - 1):
            current_effect = hypothesis.mechanism[i].effect.lower()
            next_cause = hypothesis.mechanism[i + 1].cause.lower()

            # Simple check: do they share any words?
            current_words = set(current_effect.split())
            next_words = set(next_cause.split())

            if not current_words & next_words:
                issues.append(
                    f"Mechanism steps {i+1}→{i+2} may be disconnected: "
                    f"'{hypothesis.mechanism[i].effect}' → '{hypothesis.mechanism[i+1].cause}'"
                )

        return issues

    def _check_number_sanity(
        self,
        hypothesis: GroundedHypothesis,
        baselines: list[ExtractedClaim],
    ) -> list[str]:
        """Check if predicted numbers are reasonable given baselines."""
        issues = []
        bounds = hypothesis.prediction_bounds

        if not bounds:
            return issues

        # Find relevant baseline
        metric = bounds.metric.lower()
        relevant_baselines = [
            c for c in baselines
            if c.quantitative_data and metric in c.quantitative_data.metric.lower()
        ]

        for baseline in relevant_baselines:
            baseline_value = baseline.quantitative_data.value

            # Check if prediction is within reasonable range (2x of baseline)
            if bounds.upper_bound > baseline_value * 3:
                issues.append(
                    f"Upper bound ({bounds.upper_bound}) exceeds 3x baseline "
                    f"({baseline_value}) — may be unrealistic"
                )

        return issues

    def _check_specificity(self, text: str) -> list[str]:
        """Check for vague language."""
        issues = []
        text_lower = text.lower()

        # Check for vague words
        found_vague = [
            word for word in self.VAGUE_WORDS
            if word in text_lower
        ]

        if found_vague:
            issues.append(f"Vague language detected: {', '.join(found_vague[:3])}")

        # Check for mechanism-free claims without follow-up
        for phrase in self.MECHANISM_FREE:
            if phrase in text_lower and " because " not in text_lower:
                issues.append(f"Claim uses '{phrase}' without explaining mechanism")
                break

        # Check length — too short is usually too vague
        if len(text) < 50:
            issues.append("Claim too brief — likely lacks specificity")

        return issues

    def batch_validate(
        self,
        hypotheses: list[GroundedHypothesis],
        baselines: list[ExtractedClaim] | None = None,
    ) -> list[ValidationResult]:
        """Validate a batch of hypotheses."""
        return [
            self.validate_grounded(h, baselines)
            for h in hypotheses
        ]

    def filter_valid(
        self,
        hypotheses: list[GroundedHypothesis],
        min_score: float = 0.5,
    ) -> list[GroundedHypothesis]:
        """Filter to keep only hypotheses above quality threshold."""
        valid = []
        for h in hypotheses:
            result = self.validate_grounded(h)
            if result.score >= min_score:
                valid.append(h)
        return valid
