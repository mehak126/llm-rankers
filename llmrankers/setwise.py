import copy
import random
import re
import time
from collections import Counter
from typing import List

import openai
import tiktoken
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from .rankers import LlmRanker, SearchResult

random.seed(929)


class SetwiseLlmRanker(LlmRanker):
    CHARACTERS = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
    ]  # "Passage X" and "Passage Y" will be tokenized into 3 tokens, so we dont use for now

    def __init__(
        self,
        model_name_or_path,
        tokenizer_name_or_path,
        device,
        num_child=3,
        k=10,
        scoring="generation",
        method="heapsort",
        num_permutation=1,
        cache_dir=None,
    ):

        self.device = device
        self.num_child = num_child
        self.num_permutation = num_permutation
        self.k = k
        self.config = AutoConfig.from_pretrained(
            model_name_or_path, cache_dir=cache_dir
        )
        if self.config.model_type == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained(
                (
                    tokenizer_name_or_path
                    if tokenizer_name_or_path is not None
                    else model_name_or_path
                ),
                cache_dir=cache_dir,
            )
            self.llm = T5ForConditionalGeneration.from_pretrained(
                model_name_or_path,
                #   device_map='auto',
                device_map={"": device},
                torch_dtype=torch.float16 if "cuda" in device else torch.float32,
                cache_dir=cache_dir,
            )
            self.decoder_input_ids = (
                self.tokenizer.encode(
                    "<pad> Passage", return_tensors="pt", add_special_tokens=False
                ).to(self.device)
                if self.tokenizer
                else None
            )

            self.target_token_ids = self.tokenizer.batch_encode_plus(
                [
                    f"<pad> Passage {self.CHARACTERS[i]}"
                    for i in range(len(self.CHARACTERS))
                ],
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
            ).input_ids[:, -1]
        elif self.config.model_type == "llama":
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, cache_dir=cache_dir
            )
            self.tokenizer.use_default_system_prompt = False
            if "vicuna" and "v1.5" in model_name_or_path:
                self.tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message }}{% endif %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}"
            elif "llama" in model_name_or_path.lower():
                chat_template = open(
                    "./chat_templates/chat_templates/llama-3-instruct.jinja"
                ).read()
                chat_template = chat_template.replace("    ", "").replace("\n", "")
                self.tokenizer.chat_template = chat_template
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map={"": device},
                torch_dtype=torch.float16 if "cuda" in device else torch.float32,
                cache_dir=cache_dir,
            ).eval()

        else:
            raise NotImplementedError(
                f"Model type {self.config.model_type} is not supported yet for setwise:("
            )

        self.scoring = scoring
        self.method = method
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

    def compare(self, query: str, docs: List):
        self.total_compare += 1 if self.num_permutation == 1 else self.num_permutation

        passages = "\n\n".join(
            [
                f'Passage {self.CHARACTERS[i]}: "{doc.text}"'
                for i, doc in enumerate(docs)
            ]
        )
        input_text = (
            f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n'
            + passages
            + "\n\nOutput only the passage label of the most relevant passage:"
        )

        if self.scoring == "generation":
            if self.config.model_type == "t5":

                if self.num_permutation == 1:
                    input_ids = self.tokenizer(
                        input_text, return_tensors="pt"
                    ).input_ids.to(self.device)
                    self.total_prompt_tokens += input_ids.shape[1]

                    output_ids = self.llm.generate(
                        input_ids,
                        decoder_input_ids=self.decoder_input_ids,
                        max_new_tokens=2,
                    )[0]

                    self.total_completion_tokens += output_ids.shape[0]

                    output = self.tokenizer.decode(
                        output_ids, skip_special_tokens=True
                    ).strip()
                    output = output[-1]
                else:
                    id_passage = [(i, p) for i, p in enumerate(docs)]
                    labels = [self.CHARACTERS[i] for i in range(len(docs))]
                    batch_data = []
                    for _ in range(self.num_permutation):
                        batch_data.append(
                            [
                                random.sample(id_passage, len(id_passage)),
                                random.sample(labels, len(labels)),
                            ]
                        )

                    batch_ref = []
                    input_text = []
                    for batch in batch_data:
                        ref = []
                        passages = []
                        characters = []
                        for p, c in zip(batch[0], batch[1]):
                            ref.append(p[0])
                            passages.append(p[1].text)
                            characters.append(c)
                        batch_ref.append((ref, characters))
                        passages = "\n\n".join(
                            [
                                f'Passage {characters[i]}: "{passages[i]}"'
                                for i in range(len(passages))
                            ]
                        )
                        input_text.append(
                            f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n'
                            + passages
                            + "\n\nOutput only the passage label of the most relevant passage:"
                        )

                    input_ids = self.tokenizer(
                        input_text, return_tensors="pt"
                    ).input_ids.to(self.device)
                    self.total_prompt_tokens += input_ids.shape[1] * input_ids.shape[0]

                    output_ids = self.llm.generate(
                        input_ids,
                        decoder_input_ids=self.decoder_input_ids.repeat(
                            input_ids.shape[0], 1
                        ),
                        max_new_tokens=2,
                    )
                    output = self.tokenizer.batch_decode(
                        output_ids[:, self.decoder_input_ids.shape[1] :],
                        skip_special_tokens=True,
                    )

                    # vote
                    candidates = []
                    for ref, result in zip(batch_ref, output):
                        result = result.strip().upper()
                        docids, characters = ref
                        if len(result) != 1 or result not in characters:
                            print(f"Unexpected output: {result}")
                            continue
                        win_doc = docids[characters.index(result)]
                        candidates.append(win_doc)

                    if len(candidates) == 0:
                        print(f"Unexpected voting: {output}")
                        output = "Unexpected voting."
                    else:
                        # handle tie
                        candidate_counts = Counter(candidates)
                        max_count = max(candidate_counts.values())
                        most_common_candidates = [
                            candidate
                            for candidate, count in candidate_counts.items()
                            if count == max_count
                        ]
                        if len(most_common_candidates) == 1:
                            output = self.CHARACTERS[most_common_candidates[0]]
                        else:
                            output = self.CHARACTERS[
                                random.choice(most_common_candidates)
                            ]

            elif self.config.model_type == "llama":
                conversation = [{"role": "user", "content": input_text}]
                # self.tokenizer.chat_template = (
                #     '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = "26 Jul 2024" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- "Tools: " + builtin_tools | reject(\'equalto\', \'code_interpreter\') | join(", ") + "\\n\\n"}}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] | trim + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n            {{- "<|python_tag|>" + tool_call.name + ".call(" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + \'="\' + arg_val + \'"\' }}\n                {%- if not loop.last %}\n                    {{- ", " }}\n                {%- endif %}\n                {%- endfor %}\n            {{- ")" }}\n        {%- else  %}\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n            {{- \'{"name": "\' + tool_call.name + \'", \' }}\n            {{- \'"parameters": \' }}\n            {{- tool_call.arguments | tojson }}\n            {{- "}" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we\'re in ipython mode #}\n            {{- "<|eom_id|>" }}\n        {%- else %}\n            {{- "<|eot_id|>" }}\n        {%- endif %}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n',
                # )
                # chat_template = """{{ bos_token }}
                # {%- if messages[0]['role'] == 'system' %}
                #     {%- set system_message = messages[0]['content']|trim %}
                #     {%- set messages = messages[1:] %}
                # {%- else %}
                #     {%- set system_message = "" %}
                # {%- endif %}

                # {%- if system_message %}
                # <|start_header_id|>system<|end_header_id|>
                # {{ system_message }}
                # <|eot_id|>
                # {%- endif %}

                # {%- for message in messages %}
                # <|start_header_id|>{{ message['role'] }}<|end_header_id|>
                # {{ message['content']|trim }}
                # <|eot_id|>
                # {%- endfor %}

                # {%- if add_generation_prompt %}
                # <|start_header_id|>assistant<|end_header_id|>
                # {%- endif %}"""

                # chat_template = """{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}",
                # "clean_up_tokenization_spaces": false,
                # "eos_token": {
                #     "__type": "AddedToken",
                #     "content": "</s>",
                #     "lstrip": false,
                #     "normalized": false,
                #     "rstrip": false,
                #     "single_word": false
                # },
                # "legacy": false,
                # "model_max_length": 1000000000000000019884624838656,
                # "pad_token": null,
                # "padding_side": "right",
                # "sp_model_kwargs": {},
                # "tokenizer_class": "LlamaTokenizer",
                # "unk_token": {
                #     "__type": "AddedToken",
                #     "content": "<unk>",
                #     "lstrip": false,
                #     "normalized": false,
                #     "rstrip": false,
                #     "single_word": false
                # }
                # }"""

                # self.tokenizer.chat_template = chat_template
                # if self.tokenizer.pad_token_id is None:
                #     self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

                prompt = self.tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
                prompt += " Passage:"

                inputs = self.tokenizer(
                    prompt, return_tensors="pt", padding=True, truncation=True
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask", None)

                # old code
                # input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
                # self.device
                # )

                if attention_mask is None:
                    attention_mask = [1] * len(input_ids["input_ids"])

                self.total_prompt_tokens += input_ids.shape[1]

                output_ids = self.llm.generate(
                    input_ids,
                    do_sample=False,
                    temperature=0.0,
                    top_p=None,
                    max_new_tokens=1,
                    attention_mask=attention_mask,
                )[0]

                self.total_completion_tokens += output_ids.shape[0]

                output = (
                    self.tokenizer.decode(
                        output_ids[input_ids.shape[1] :], skip_special_tokens=True
                    )
                    .strip()
                    .upper()
                )

        elif self.scoring == "likelihood":
            if self.config.model_type == "t5":
                input_ids = self.tokenizer(
                    input_text, return_tensors="pt"
                ).input_ids.to(self.device)
                self.total_prompt_tokens += input_ids.shape[1]
                with torch.no_grad():
                    logits = self.llm(
                        input_ids=input_ids, decoder_input_ids=self.decoder_input_ids
                    ).logits[0][-1]
                    distributions = torch.softmax(logits, dim=0)
                    scores = distributions[self.target_token_ids[: len(docs)]]
                    ranked = sorted(
                        zip(self.CHARACTERS[: len(docs)], scores),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    output = ranked[0][0]

            else:
                raise NotImplementedError

        if len(output) == 1 and output in self.CHARACTERS:
            pass
        else:
            print(f"Unexpected output: {output}")

        return output

    def heapify(self, arr, n, i, query):
        # Find largest among root and children
        if self.num_child * i + 1 < n:  # if there are children
            docs = [arr[i]] + arr[
                self.num_child * i + 1 : min((self.num_child * (i + 1) + 1), n)
            ]
            inds = [i] + list(
                range(self.num_child * i + 1, min((self.num_child * (i + 1) + 1), n))
            )
            output = self.compare(query, docs)
            try:
                best_ind = self.CHARACTERS.index(output)
            except ValueError:
                best_ind = 0
            try:
                largest = inds[best_ind]
            except IndexError:
                largest = i
            # If root is not largest, swap with largest and continue heapifying
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                self.heapify(arr, n, largest, query)

    def heapSort(self, arr, query, k):
        n = len(arr)
        ranked = 0
        # Build max heap
        for i in range(n // self.num_child, -1, -1):
            self.heapify(arr, n, i, query)
        for i in range(n - 1, 0, -1):
            # Swap
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked == k:
                break
            # Heapify root element
            self.heapify(arr, i, 0, query)

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        original_ranking = copy.deepcopy(ranking)
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

        if self.method == "heapsort":
            self.heapSort(ranking, query, self.k)
            ranking = list(reversed(ranking))
        elif self.method == "bubblesort":
            last_start = len(ranking) - (self.num_child + 1)

            for i in range(self.k):
                start_ind = last_start
                end_ind = last_start + (self.num_child + 1)
                is_change = False
                while True:
                    if start_ind < i:
                        start_ind = i
                    output = self.compare(query, ranking[start_ind:end_ind])
                    try:
                        best_ind = self.CHARACTERS.index(output)
                    except ValueError:
                        best_ind = 0
                    if best_ind != 0:
                        ranking[start_ind], ranking[start_ind + best_ind] = (
                            ranking[start_ind + best_ind],
                            ranking[start_ind],
                        )
                        if not is_change:
                            is_change = True
                            if (
                                last_start != len(ranking) - (self.num_child + 1)
                                and best_ind == len(ranking[start_ind:end_ind]) - 1
                            ):
                                last_start += len(ranking[start_ind:end_ind]) - 1

                    if start_ind == i:
                        break

                    if not is_change:
                        last_start -= self.num_child

                    start_ind -= self.num_child
                    end_ind -= self.num_child

        ##  this is a bit slower but standard bobblesort implementation, keep here FYI
        # elif self.method == "bubblesort":
        #     for i in range(k):
        #         start_ind = len(ranking) - (self.num_child + 1)
        #         end_ind = len(ranking)
        #         while True:
        #             if start_ind < i:
        #                 start_ind = i
        #             output = self.compare(query, ranking[start_ind:end_ind])
        #             try:
        #                 best_ind = self.CHARACTERS.index(output)
        #             except ValueError:
        #                 best_ind = 0
        #             if best_ind != 0:
        #                 ranking[start_ind], ranking[start_ind + best_ind] = ranking[start_ind + best_ind], ranking[start_ind]
        #
        #             if start_ind == i:
        #                 break
        #
        #             start_ind -= self.num_child
        #             end_ind -= self.num_child

        else:
            raise NotImplementedError(f"Method {self.method} is not implemented.")

        results = []
        top_doc_ids = set()
        rank = 1

        for i, doc in enumerate(ranking[: self.k]):
            top_doc_ids.add(doc.docid)
            results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
            rank += 1
        for doc in original_ranking:
            if doc.docid not in top_doc_ids:
                results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
                rank += 1

        return results

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(
            self.tokenizer.tokenize(text)[:length]
        )


class OpenAiSetwiseLlmRanker(SetwiseLlmRanker):
    def __init__(
        self, model_name_or_path, api_key, num_child=3, method="heapsort", k=10
    ):
        self.llm = model_name_or_path
        self.tokenizer = tiktoken.encoding_for_model(model_name_or_path)
        self.num_child = num_child
        self.method = method
        self.k = k
        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.system_prompt = "You are RankGPT, an intelligent assistant specialized in selecting the most relevant passage from a pool of passages based on their relevance to the query."
        openai.api_key = api_key

    def compare(self, query: str, docs: List):
        self.total_compare += 1
        passages = "\n\n".join(
            [
                f'Passage {self.CHARACTERS[i]}: "{doc.text}"'
                for i, doc in enumerate(docs)
            ]
        )
        input_text = (
            f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n'
            + passages
            + "\n\nOutput only the passage label of the most relevant passage."
        )

        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=self.llm,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": input_text},
                    ],
                    temperature=0.0,
                    request_timeout=15,
                )

                self.total_completion_tokens += int(
                    response["usage"]["completion_tokens"]
                )
                self.total_prompt_tokens += int(response["usage"]["prompt_tokens"])

                output = response["choices"][0]["message"]["content"]
                matches = re.findall(r"(Passage [A-Z])", output, re.MULTILINE)
                if matches:
                    output = matches[0][8]
                elif output.strip() in self.CHARACTERS:
                    pass
                else:
                    print(f"Unexpected output: {output}")
                    output = "A"
                return output

            except openai.error.APIError as e:
                # Handle API error here, e.g. retry or log
                print(f"OpenAI API returned an API Error: {e}")
                time.sleep(5)
                continue
            except openai.error.APIConnectionError as e:
                # Handle connection error here
                print(f"Failed to connect to OpenAI API: {e}")
                time.sleep(5)
                continue
            except openai.error.RateLimitError as e:
                # Handle rate limit error (we recommend using exponential backoff)
                print(f"OpenAI API request exceeded rate limit: {e}")
                time.sleep(5)
                continue
            except openai.error.InvalidRequestError as e:
                # Handle invalid request error
                print(f"OpenAI API request was invalid: {e}")
                raise e
            except openai.error.AuthenticationError as e:
                # Handle authentication error
                print(f"OpenAI API request failed authentication: {e}")
                raise e
            except openai.error.Timeout as e:
                # Handle timeout error
                print(f"OpenAI API request timed out: {e}")
                time.sleep(5)
                continue
            except openai.error.ServiceUnavailableError as e:
                # Handle service unavailable error
                print(
                    f"OpenAI API request failed with a service unavailable error: {e}"
                )
                time.sleep(5)
                continue
            except Exception as e:
                print(f"Unknown error: {e}")
                raise e

    def truncate(self, text, length):
        return self.tokenizer.decode(self.tokenizer.encode(text)[:length])
