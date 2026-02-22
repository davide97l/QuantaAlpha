from quantaalpha.coder.costeer.evaluators import CoSTEERSingleFeedback
from quantaalpha.coder.costeer.evolvable_subjects import EvolvingItem
from quantaalpha.core.evolving_agent import RAGEvoAgent
from quantaalpha.core.evolving_framework import EvolvableSubjects


class FilterFailedRAGEvoAgent(RAGEvoAgent):
    def filter_evolvable_subjects_by_feedback(
        self, evo: EvolvableSubjects, feedback: CoSTEERSingleFeedback
    ) -> EvolvableSubjects:
        assert isinstance(evo, EvolvingItem)
        assert isinstance(feedback, list)
        assert len(evo.sub_workspace_list) == len(feedback)

        # NOTE: Do NOT clear workspaces here even when final_decision=False.
        # This method is called after the final CoSTEER iteration.  Clearing the
        # workspace at this point deletes factor.py and empties code_dict, which
        # causes process_factor_data() to find no executable code.
        # Workspaces are only cleared mid-loop (see multistep_evolve) to free
        # space before the next iteration; on the final pass we want to keep
        # every workspace that produced any code so the runner can evaluate it.
        return evo
