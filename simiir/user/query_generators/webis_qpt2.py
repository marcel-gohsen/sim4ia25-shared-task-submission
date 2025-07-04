from simiir.user.query_generators.base import BaseQueryGenerator
from ifind.common.utils import get_given_queries

import random
from transformers import AutoModelForCausalLM, AutoTokenizer


class WebisQPT2_A1(BaseQueryGenerator):
    """
    Takes the first given query, and turns it into a question.
    """

    def __init__(self, stopword_file, query_file, user, background_file=[]):
        super().__init__(stopword_file, background_file=[])
        self.__query_filename = query_file
        self.__user = user

        model_repo = "marcel-gohsen/qpt2-aql-aol-sim4ia-inst"
        self.model = AutoModelForCausalLM.from_pretrained(model_repo)
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo)

    def _predict_queries(self, queries) -> list[tuple[str, int]]:
        prompt = self.tokenizer.sep_token.join(queries) + self.tokenizer.sep_token
        tokens = self.tokenizer(prompt, return_tensors="pt")

        output = self.model.generate(**tokens,
                                     num_return_sequences=10,
                                     num_beams=10,
                                     do_sample=False,
                                     num_beam_groups=10,
                                diversity_penalty=1.5)

        output = output[:, tokens.input_ids.shape[1]:]

        orig_query = [q.lower() for q in queries]

        out_queries = set()
        for beam_outputs in output:
            out = self.tokenizer.decode(beam_outputs, skip_special_tokens=False)

            query = out.split(self.tokenizer.sep_token)[0]
            query = query.replace(self.tokenizer.eos_token, '')

            if len(query) > 0 and query not in orig_query:
                out_queries.add(query)

        return [(x, i) for i, x in enumerate(out_queries)]

    def generate_query_list(self, user_context) -> list[tuple[str, int]]:
        return_queries: list[tuple[str, int]] = get_given_queries(
            self.__query_filename, self.__user, user_context.topic.id, task_a2=False
        )

        last_query: str = return_queries[-1][0]
        new_queries: list[tuple[str, int]] = self._predict_queries([last_query])

        return_queries.extend(new_queries)

        return return_queries


class WebisQPT2_A2(BaseQueryGenerator):
    """
    Takes the first given query, and turns it into a question.
    """

    def __init__(self, stopword_file, query_file, user, background_file=[]):
        super().__init__(stopword_file, background_file=[])
        self.__query_filename = query_file
        self.__user = user

        model_repo = "marcel-gohsen/qpt2-aql-aol-sim4ia-inst"
        self.model = AutoModelForCausalLM.from_pretrained(model_repo)
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo)

    def _predict_queries(self, queries) -> list[tuple[str, int]]:
        prompt = self.tokenizer.sep_token.join(queries) + self.tokenizer.sep_token
        tokens = self.tokenizer(prompt, return_tensors="pt")

        output = self.model.generate(**tokens,
                                     num_return_sequences=10,
                                     num_beams=10,
                                     do_sample=False,
                                     num_beam_groups=10,
                                diversity_penalty=1.5)

        output = output[:, tokens.input_ids.shape[1]:]

        orig_query = [q.lower() for q in queries]

        out_queries = set()
        for beam_outputs in output:
            out = self.tokenizer.decode(beam_outputs, skip_special_tokens=False)

            query = out.split(self.tokenizer.sep_token)[0]
            query = query.replace(self.tokenizer.eos_token, '')

            if len(query) > 0 and query not in orig_query:
                out_queries.add(query)

        return [(x, i) for i, x in enumerate(out_queries)]

    def generate_query_list(self, user_context) -> list[tuple[str, int]]:
        return_queries: list[tuple[str, int]] = get_given_queries(
            self.__query_filename, self.__user, user_context.topic.id, task_a2=False
        )

        queries: list[str] = [q[0] for q in return_queries]
        new_queries: list[tuple[str, int]] = self._predict_queries(queries)

        return_queries.extend(new_queries)

        return return_queries


class WebisQPT2Medium_A1(BaseQueryGenerator):
    """
    Takes the first given query, and turns it into a question.
    """

    def __init__(self, stopword_file, query_file, user, background_file=[]):
        super().__init__(stopword_file, background_file=[])
        self.__query_filename = query_file
        self.__user = user

        model_repo = "marcel-gohsen/qpt2-medium-aql-aol-sim4ia-inst"
        self.model = AutoModelForCausalLM.from_pretrained(model_repo)
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo)

    def _predict_queries(self, queries) -> list[tuple[str, int]]:
        prompt = self.tokenizer.sep_token.join(queries) + self.tokenizer.sep_token
        tokens = self.tokenizer(prompt, return_tensors="pt")

        output = self.model.generate(**tokens,
                                     num_return_sequences=10,
                                     num_beams=10,
                                     do_sample=False,
                                     num_beam_groups=10,
                                diversity_penalty=1.5)

        output = output[:, tokens.input_ids.shape[1]:]

        orig_query = [q.lower() for q in queries]

        out_queries = set()
        for beam_outputs in output:
            out = self.tokenizer.decode(beam_outputs, skip_special_tokens=False)

            query = out.split(self.tokenizer.sep_token)[0]
            query = query.replace(self.tokenizer.eos_token, '')

            if len(query) > 0 and query not in orig_query:
                out_queries.add(query)

        return [(x, i) for i, x in enumerate(out_queries)]

    def generate_query_list(self, user_context) -> list[tuple[str, int]]:
        return_queries: list[tuple[str, int]] = get_given_queries(
            self.__query_filename, self.__user, user_context.topic.id, task_a2=False
        )

        last_query: str = return_queries[-1][0]
        new_queries: list[tuple[str, int]] = self._predict_queries([last_query])

        return_queries.extend(new_queries)

        return return_queries


class WebisQPT2Medium_A2(BaseQueryGenerator):
    """
    Takes the first given query, and turns it into a question.
    """

    def __init__(self, stopword_file, query_file, user, background_file=[]):
        super().__init__(stopword_file, background_file=[])
        self.__query_filename = query_file
        self.__user = user

        model_repo = "marcel-gohsen/qpt2-medium-aql-aol-sim4ia-inst"
        self.model = AutoModelForCausalLM.from_pretrained(model_repo)
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo)

    def _predict_queries(self, queries) -> list[tuple[str, int]]:
        prompt = self.tokenizer.sep_token.join(queries) + self.tokenizer.sep_token
        tokens = self.tokenizer(prompt, return_tensors="pt")

        output = self.model.generate(**tokens,
                                     num_return_sequences=10,
                                     num_beams=10,
                                     do_sample=False,
                                     num_beam_groups=10,
                                diversity_penalty=1.5)

        output = output[:, tokens.input_ids.shape[1]:]

        orig_query = [q.lower() for q in queries]

        out_queries = set()
        for beam_outputs in output:
            out = self.tokenizer.decode(beam_outputs, skip_special_tokens=False)

            query = out.split(self.tokenizer.sep_token)[0]
            query = query.replace(self.tokenizer.eos_token, '')

            if len(query) > 0 and query not in orig_query:
                out_queries.add(query)

        return [(x, i) for i, x in enumerate(out_queries)]

    def generate_query_list(self, user_context) -> list[tuple[str, int]]:
        return_queries: list[tuple[str, int]] = get_given_queries(
            self.__query_filename, self.__user, user_context.topic.id, task_a2=False
        )

        queries: list[str] = [q[0] for q in return_queries]
        new_queries: list[tuple[str, int]] = self._predict_queries(queries)

        return_queries.extend(new_queries)

        return return_queries


class WebisGPT2_A1(BaseQueryGenerator):
    """
    Takes the first given query, and turns it into a question.
    """

    def __init__(self, stopword_file, query_file, user, background_file=[]):
        super().__init__(stopword_file, background_file=[])
        self.__query_filename = query_file
        self.__user = user

        model_repo = "marcel-gohsen/gpt2-aql-aol-sim4ia-inst"
        self.model = AutoModelForCausalLM.from_pretrained(model_repo)
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo)

    def _predict_queries(self, queries) -> list[tuple[str, int]]:
        prompt = self.tokenizer.sep_token.join(queries) + self.tokenizer.sep_token
        tokens = self.tokenizer(prompt, return_tensors="pt")

        output = self.model.generate(**tokens,
                                     num_return_sequences=10,
                                     num_beams=10,
                                     do_sample=False,
                                     num_beam_groups=10,
                                diversity_penalty=1.5)

        output = output[:, tokens.input_ids.shape[1]:]

        orig_query = [q.lower() for q in queries]

        out_queries = set()
        for beam_outputs in output:
            out = self.tokenizer.decode(beam_outputs, skip_special_tokens=False)

            query = out.split(self.tokenizer.sep_token)[0]
            query = query.replace(self.tokenizer.eos_token, '')

            if len(query) > 0 and query not in orig_query:
                out_queries.add(query)

        return [(x, i) for i, x in enumerate(out_queries)]

    def generate_query_list(self, user_context) -> list[tuple[str, int]]:
        return_queries: list[tuple[str, int]] = get_given_queries(
            self.__query_filename, self.__user, user_context.topic.id, task_a2=False
        )

        last_query: str = return_queries[-1][0]
        new_queries: list[tuple[str, int]] = self._predict_queries([last_query])

        return_queries.extend(new_queries)

        return return_queries


class WebisGPT2_A2(BaseQueryGenerator):
    """
    Takes the first given query, and turns it into a question.
    """

    def __init__(self, stopword_file, query_file, user, background_file=[]):
        super().__init__(stopword_file, background_file=[])
        self.__query_filename = query_file
        self.__user = user

        model_repo = "marcel-gohsen/gpt2-aql-aol-sim4ia-inst"
        self.model = AutoModelForCausalLM.from_pretrained(model_repo)
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo)

    def _predict_queries(self, queries) -> list[tuple[str, int]]:
        prompt = self.tokenizer.sep_token.join(queries) + self.tokenizer.sep_token
        tokens = self.tokenizer(prompt, return_tensors="pt")

        output = self.model.generate(**tokens,
                                     num_return_sequences=10,
                                     num_beams=10,
                                     do_sample=False,
                                     num_beam_groups=10,
                                diversity_penalty=1.5)

        output = output[:, tokens.input_ids.shape[1]:]

        orig_query = [q.lower() for q in queries]

        out_queries = set()
        for beam_outputs in output:
            out = self.tokenizer.decode(beam_outputs, skip_special_tokens=False)

            query = out.split(self.tokenizer.sep_token)[0]
            query = query.replace(self.tokenizer.eos_token, '')

            if len(query) > 0 and query not in orig_query:
                out_queries.add(query)

        return [(x, i) for i, x in enumerate(out_queries)]

    def generate_query_list(self, user_context) -> list[tuple[str, int]]:
        return_queries: list[tuple[str, int]] = get_given_queries(
            self.__query_filename, self.__user, user_context.topic.id, task_a2=False
        )

        queries: list[str] = [q[0] for q in return_queries]
        new_queries: list[tuple[str, int]] = self._predict_queries(queries)

        return_queries.extend(new_queries)

        return return_queries

