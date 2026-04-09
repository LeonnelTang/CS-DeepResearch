# Copyright (c) ModelScope Contributors. All rights reserved.
# Multi-turn Rollout Schedulers for GRPO training.
import asyncio
from abc import ABC
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

from swift.infer_engine import GRPOVllmEngine
from swift.infer_engine.protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, RequestConfig,
                                         RolloutInferRequest, RolloutOutput)
from swift.template import Messages
from swift.utils import remove_response
from .gym_env import ContextManager, Env, context_managers, envs

from myRL.tools.base import BaseTool
from myRL.tools.tool_config import ToolConfig
from myRL.core.parser import LLMOutputParser
from myRL.core.memory import StateManager
from openai import OpenAI
import sys
import os
import asyncio
from abc import ABC
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import logging

class RolloutScheduler(ABC):
    # Single Turn Rollout Scheduler
    def __init__(self, infer_engine: Optional[GRPOVllmEngine] = None, max_turns: Optional[int] = None, *args, **kwargs):
        print(">>> [DEBUG] 成功进入 RolloutScheduler.__init__", flush=True) 

        self.infer_engine = infer_engine
        # Tokenizer can be passed explicitly (e.g., in colocate mode where infer_engine may be None)
        self._tokenizer = kwargs.get('tokenizer', None)
        self.max_turns = max_turns

        print(">>> [DEBUG] 成功完成 RolloutScheduler.__init__", flush=True)

    async def async_infer(self,
                          infer_requests: List[Union['RolloutInferRequest', Dict[str, Any]]],
                          request_config: 'RequestConfig',
                          *,
                          use_tqdm: Optional[bool] = None,
                          **kwargs) -> List['RolloutOutput']:
        """
        Perform asynchronous batched inference for multiple rollout requests.

        This method serves as the main entry point for multi-round training inference.
        It executes the `run` method for each inference request concurrently and
        aggregates the results into a single flattened list.

        Each inference request can be either a `RolloutInferRequest` instance or a
        dictionary that can be converted into one. The results from all requests are
        collected asynchronously using the underlying inference engine.

        Args:
            infer_requests (List[Union[RolloutInferRequest, Dict[str, Any]]]):
                A list of inference requests. Each request can be either:
                    - A `RolloutInferRequest` object.
                    - A dictionary containing the fields required to initialize a
                    `RolloutInferRequest`.
            request_config (RequestConfig):
                Configuration object specifying inference settings. Must satisfy
                `request_config.n == 1`, as only single-response generation is supported.
            use_tqdm (Optional[bool], optional):
                Whether to display a progress bar during batch inference.
                If `None`, it defaults to `True` when there are multiple requests,
                otherwise `False`.
            **kwargs:
                Additional arguments forwarded to the underlying `run` method.

        Returns:
            List[RolloutOutput]:
                A list of RolloutOutput objects corresponding to the provided inference requests.

        Raises:
            AssertionError:
                If `request_config.n` is not equal to `1`.

        Notes:
            - Internally, this method converts dict-based requests into
            `RolloutInferRequest` instances.
            - Uses `infer_engine._batch_infer_stream` to perform concurrent execution.
            - The returned list is guaranteed to be flattened, even if individual
            tasks return lists of responses.
        """

        assert request_config.n == 1

        async def _infer_async_single(infer_request: Union['RolloutInferRequest', Dict[str, Any]],
                                      request_config: 'RequestConfig', **kwargs):
            if isinstance(infer_request, Dict):
                infer_request = RolloutInferRequest(**infer_request)

            return await self.run(infer_request, request_config, **kwargs)

        tasks = [_infer_async_single(infer_request, request_config, **kwargs) for infer_request in infer_requests]
        if use_tqdm is None:
            use_tqdm = len(infer_requests) > 1
        # Execute all tasks and flatten the results
        results = await self.infer_engine._batch_infer_stream(tasks, request_config.stream, use_tqdm, None)
        # Flatten the results since each task may return a list
        flattened_results = []
        for result in results:
            if isinstance(result, list):
                flattened_results.extend(result)
            else:
                flattened_results.append(result)
        return flattened_results

    async def run(self, infer_request: 'RolloutInferRequest', request_config: 'RequestConfig',
                  **kwargs) -> 'RolloutOutput':
        response: 'ChatCompletionResponse' = await self.infer_engine.infer_async(infer_request, request_config,
                                                                                 **kwargs)
        response_token_ids = response.choices[0].token_ids
        response_loss_mask = [1] * len(response_token_ids)
        return RolloutOutput(
            response=response,
            messages=infer_request.messages,
            response_token_ids=[response_token_ids],
            response_loss_mask=[response_loss_mask],
            rollout_infos={'num_turns': 1})

    def __getattr__(self, key: str):
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            pass

        try:
            infer_engine = object.__getattribute__(self, 'infer_engine')
            if hasattr(infer_engine, key):
                return getattr(infer_engine, key)
            if hasattr(infer_engine.engine, key):
                return getattr(infer_engine.engine, key)

        except AttributeError:
            raise AttributeError(f'{type(self).__name__} object has no attribute {key}')

    @property
    def engine(self):
        return self.infer_engine

    @property
    def tokenizer(self):
        """Get tokenizer, prioritizing explicitly passed tokenizer over infer_engine's tokenizer."""
        if self._tokenizer is not None:
            return self._tokenizer
        if self.infer_engine is not None:
            return self.infer_engine.tokenizer
        return None


class MultiTurnScheduler(RolloutScheduler, ABC):
    """
    Abstract base class for multi-turn rollout scheduling.

    Provides default implementation for multi-turn conversation management with two customization approaches:

    1. FULL CUSTOMIZATION:
       Override the `run()` method to implement completely custom multi-turn logic.
       - Gives full control over the rollout process
       - Must handle all turn management and termination logic

    2. PARTIAL CUSTOMIZATION:
       Implement the required `step()` method and optionally override `check_finished()`
       - Uses MultiTurnScheduler's run() method infrastructure
       - Only need to implement turn transition logic in step()
       - Optionally customize termination conditions

    Note: You must implement at least one of these approaches in your subclass.

    Options:
        - If each round's response_token_ids are included in the RolloutOutput,
          the Trainer can skip encoding the completion text into token_ids when calculating loss.
          This avoids potential training inconsistencies due to asymmetric encode/decode behavior.
          See: https://github.com/0russwest0/Agent-R1/issues/30#issuecomment-2826155367

        - If both response_token_ids and response_loss_mask are returned in the RolloutOutput,
          you can manually control the loss mask for each token.
          The Trainer will use the provided loss_mask values directly when computing the loss.
          Note: Returning response_loss_mask requires that response_token_ids are also returned,
          as the two must be aligned in length for correct loss computation.

        You can refer to MathTipsScheduler as an example of how to use response_token_ids and response_loss_mask.

    Loss mask configuration:
        During rollout, some parts of the completion (e.g., environment observations embedded in completion)
        may need to be masked out from loss computation.
        There are two supported strategies:

        1. Use the built-in `loss_scale` parameter in ms-swift and do not return response token ids.
        2. Return response_token_ids along with a corresponding response_loss_mask (of equal length) to indicate the loss mask for each token. # noqa
    """

    async def run(self, infer_request: 'RolloutInferRequest', request_config: 'RequestConfig',
                  **kwargs) -> Union['RolloutOutput', List['RolloutOutput']]:
        """Execute multi-turn conversation rollout with built-in turn management logic.

        This implements the default multi-turn interaction flow that can be overridden
        to customize conversation handling behavior. The default logic provides:

        1. Automatic conversation turn management and stopping conditions
        2. Seamless message accumulation across multiple turns
        3. Response token tracking and loss mask management
        4. Configurable early stopping mechanisms

        Args:
            infer_request: The initial inference request containing conversation messages
            request_config: Configuration parameters for the inference request
            **kwargs: Additional inference parameters passed to the engine

        Returns:
            RolloutOutput containing the complete conversation history and metadata,
            or a list of outputs for batched requests

        Customization Approaches:
            - Override check_finished() to implement custom stopping criteria
            - Override step() to customize turn-to-turn transition logic
            - Override this entire run() method for completely custom multi-turn behavior

        Important Notes:
            - Method overriding is only supported when using server mode (swift rollout)
              with vllm_use_async_engine=True
            - Custom implementations must maintain async/await compatibility
            - Ensure proper handling of conversation state across turns

        Example:
            class CustomScheduler(MultiTurnScheduler):
                async def run(self, infer_request, request_config, **kwargs):
                    # Implement custom multi-turn conversation logic
                    # Must return RolloutOutput or List[RolloutOutput]
                    ...
        """
        current_request = infer_request
        current_turn = 1
        rollout_infos = {}
        total_response_ids = []
        total_response_loss_mask = []
        total_rollout_logprobs = []
        while True:
            messages = current_request.messages
            if current_turn == 1:
                # If it's the first turn, remove the response
                # Keep the original logic, but I think this step is unnecessary here.
                remove_response(messages)

            # Get model response
            response: 'ChatCompletionResponse' = await self.infer_engine.infer_async(
                current_request, request_config, **kwargs)
            response_choice: 'ChatCompletionResponseChoice' = response.choices[0]

            if current_turn > 1 and not messages[-1]['content']:
                # The dummy assistant message was intentionally kept during `infer_async`
                # to ensure correct history processing by the template.
                # It is now removed before appending the new completion.
                # otherwise, a syntax error would occur when executing messages[-1]['content'] += completion.
                remove_response(messages)

            # Update conversation history
            completion = response_choice.message.content
            is_continuation = False
            if messages[-1]['role'] == 'assistant':
                messages[-1]['content'] += completion
                is_continuation = True
            else:
                messages.append({'role': 'assistant', 'content': completion})

            # Check stopping conditions
            should_stop = self.check_finished(current_request, response_choice, current_turn)

            # double-check if user forget to judge the max_turns
            if self.max_turns:
                should_stop = should_stop or (current_turn >= self.max_turns)

            if should_stop:
                # Collect final turn's data
                current_logprobs = self._extract_logprobs_from_choice(response_choice)
                final_token_ids = response_choice.token_ids

                if is_continuation and total_response_ids:
                    # For continuation, extend the last turn's data
                    total_response_ids[-1].extend(final_token_ids)
                    if total_response_loss_mask:
                        total_response_loss_mask[-1].extend([1] * len(final_token_ids))
                    if total_rollout_logprobs and current_logprobs:
                        total_rollout_logprobs[-1].extend(current_logprobs)
                elif not total_response_ids:
                    # First turn stopped immediately - need to initialize with final response data
                    if final_token_ids:
                        total_response_ids = [list(final_token_ids)]
                        total_response_loss_mask = [[1] * len(final_token_ids)]
                    if current_logprobs:
                        total_rollout_logprobs = [current_logprobs]

                # Validate rollout_logprobs completeness: if logprobs are incomplete (missing for some turns),
                # clear them to disable rollout importance sampling correction (which requires complete logprobs)
                # Note: rollout_logprobs should match the number of loss_mask=1 tokens, not total response tokens
                # because completion_mask in grpo_trainer is based on labels != -100, which corresponds to loss_mask=1
                final_rollout_logprobs = total_rollout_logprobs
                if total_rollout_logprobs:
                    total_logprob_count = sum(len(turn_lps) for turn_lps in total_rollout_logprobs)
                    if total_response_loss_mask:
                        # Check if the number of logprobs matches the number of loss_mask=1 tokens
                        total_loss_mask_1_count = sum(sum(mask) for mask in total_response_loss_mask)
                        if total_loss_mask_1_count != total_logprob_count:
                            # Incomplete logprobs, clear them
                            final_rollout_logprobs = []
                    else:
                        if total_response_ids:
                            total_response_id_count = sum(len(turn_ids) for turn_ids in total_response_ids)
                            if total_response_id_count != total_logprob_count:
                                final_rollout_logprobs = []
                        else:
                            final_rollout_logprobs = []

                return RolloutOutput(
                    response=response,
                    messages=messages,
                    response_token_ids=total_response_ids,
                    response_loss_mask=total_response_loss_mask,
                    rollout_infos={
                        **rollout_infos, 'num_turns': current_turn
                    },
                    rollout_logprobs=final_rollout_logprobs,
                )

            # Prepare next turn
            ret = self.step(current_request, response_choice, current_turn)
            current_request: 'RolloutInferRequest' = ret['infer_request']

            # Track response tokens and masks
            return_token_id = False
            if 'response_token_ids' in ret:
                if is_continuation and total_response_ids:
                    total_response_ids[-1].extend(ret['response_token_ids'])
                else:
                    total_response_ids.append(ret['response_token_ids'])
                return_token_id = True

            if 'response_loss_mask' in ret:
                assert return_token_id, 'You must return response_token_ids if you want to return response_loss_mask'
                assert len(ret['response_loss_mask']) == len(ret['response_token_ids']), \
                    'response_loss_mask must have the same length as response_token_ids'
                if is_continuation and total_response_loss_mask:
                    total_response_loss_mask[-1].extend(ret['response_loss_mask'])
                else:
                    total_response_loss_mask.append(ret['response_loss_mask'])

            if 'rollout_infos' in ret:
                # Always overwrite the rollout info for this step.
                # If you need to keep all step-wise details, switch to append or merge instead.
                rollout_infos.update(ret['rollout_infos'])

            # Track rollout_logprobs for rollout importance sampling correction
            # Prefer step's returned logprobs (which may be modified/truncated) over raw response_choice logprobs
            if 'rollout_logprobs' in ret and ret['rollout_logprobs']:
                current_logprobs = ret['rollout_logprobs']
            else:
                current_logprobs = self._extract_logprobs_from_choice(response_choice)
            if current_logprobs:
                if is_continuation and total_rollout_logprobs:
                    total_rollout_logprobs[-1].extend(current_logprobs)
                else:
                    total_rollout_logprobs.append(current_logprobs)

            if current_request.messages[-1]['role'] == 'assistant':
                # Add a dummy response to allow engine to continue generating
                current_request.messages.append({'role': 'assistant', 'content': None})

            current_turn += 1

    def step(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
             current_turn: int) -> Dict:
        """
        Handles transition between conversation turns.

        Args:
            infer_request: Current inference request
            response_choice: Response from current turn
            current_turn: Current turn number

        Returns:
            Dict[str, Any]: A dictionary containing inference results with the following structure:
                - infer_request (required): Main inference request object
                - response_token_ids (Optional[List[int]]): Token IDs of response for current rollout turn
                - response_loss_mask (Optional[List[int]]): Loss mask for response tokens (same length as response_token_ids) # noqa
                - rollout_logprobs (Optional[List[float]]): Log probabilities for response tokens.
                    If not provided, will be extracted from response_choice.logprobs as fallback.
                    Useful when modifying response content (e.g., adding prompts) to avoid logprob misalignment.
                - rollout_infos (Optional[Dict[str, Any]]): Additional metadata (must be serializable)

        """
        raise NotImplementedError(
            'Please implement the `step` method in your MultiTurnScheduler subclass, or override the `run` method.')

    def check_finished(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
                       current_turn: int) -> bool:
        """
        Default termination logic for checking if a multi-turn rollout should end.

        This method is invoked by:
        - The base class MultiTurnScheduler.run() method, OR
        - Custom run() methods when explicitly called

        Note: This is the default implementation that can be overridden by subclasses for custom termination logic.

        Termination Conditions:
        1. When response hits length limit (finish_reason == 'length')
        2. When conversation reaches max_turns (if max_turns is set)

        Args:
            infer_request: The inference request object
            response_choice: Contains generation results including finish_reason
            current_turn: Current conversation turn count

        Returns:
            bool: True to terminate conversation, False to continue
        """
        if response_choice.finish_reason == 'length':
            return True
        if self.max_turns and current_turn >= self.max_turns:
            return True
        return False

    @staticmethod
    def _extract_logprobs_from_choice(response_choice: 'ChatCompletionResponseChoice') -> List[float]:
        """Extract logprobs list from response choice for rollout importance sampling.

        Args:
            response_choice: The response choice containing logprobs

        Returns:
            List of logprob values, or empty list if not available
        """
        if response_choice.logprobs is None:
            return []
        if 'content' in response_choice.logprobs:
            return [item['logprob'] for item in response_choice.logprobs['content']]
        return []


class ThinkingModelTipsScheduler(MultiTurnScheduler):
    """
    Scheduler for multi-turn reasoning with Thinking class models.

    Key Features:
    1. Parses both "think" and "answer" content from each assistant response.
    2. For each round, only the "think" content from the last round is retained in the message history.
    3. Each round's conversation history is processed independently.
    4. Returns a list of RolloutOutput objects, one for each round.
    5. Please set `--loss_scale last_round` for training last round response.

    The scheduler will automatically inject a tip prompt if the answer is incorrect, encouraging the model to recheck its reasoning. # noqa
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        acc_func = kwargs.get('acc_function', None)
        if acc_func is None:
            from swift.rewards.orm import MathAccuracy
            acc_func = MathAccuracy()
        self.acc_func = acc_func
        self.tips_prompt = 'The answer is not correct, It seems You made a mistake, you need to recheck very carefully.'

    async def run(self, infer_request: 'RolloutInferRequest', request_config: 'RequestConfig',
                  **kwargs) -> List['RolloutOutput']:
        """
        Execute multi-turn inference for Thinking models.

        Args:
            infer_request (RolloutInferRequest): The initial inference request containing the conversation history.
            request_config (RequestConfig): Configuration for the inference request.
            **kwargs: Additional arguments for the inference engine.

        Returns:
            List[RolloutOutput]: A list of RolloutOutput objects, one for each reasoning round.
        """
        current_request = infer_request
        current_turn = 1
        rollout_outputs = []

        while True:
            messages = current_request.messages
            # Obtain model response for the current turn
            response: 'ChatCompletionResponse' = await self.infer_engine.infer_async(
                current_request, request_config, **kwargs)
            response_choice: 'ChatCompletionResponseChoice' = response.choices[0]
            completion = response_choice.message.content

            # Append the assistant's response to the message history
            messages.append({'role': 'assistant', 'content': completion})

            # Construct the message history for this round, keeping only the last "think" content
            messages_with_last_think = self._build_messages(messages)

            # Create a RolloutOutput for the current round
            round_output = RolloutOutput(
                response=response,
                messages=messages_with_last_think,
                response_token_ids=response_choice.token_ids,
                rollout_infos={'num_turns': current_turn})
            # Store the output for this round
            rollout_outputs.append(round_output)

            # Determine whether to stop the multi-turn reasoning
            should_stop = self.check_finished(current_request, response_choice, current_turn)

            if should_stop:
                break

            # Prepare for the next turn by updating the inference request
            ret = self.step(current_request, response_choice, current_turn)
            current_request: 'RolloutInferRequest' = ret['infer_request']
            current_turn += 1

        return rollout_outputs

    def check_finished(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
                       current_turn: int) -> bool:

        last_query = infer_request.messages[-2]['content']
        # tips once
        if self.tips_prompt in last_query:
            return True

        completion = response_choice.message.content
        solution = infer_request.data_dict['solution']
        acc = self.acc_func([completion], [solution])[0]
        if acc == 1:
            return True

        return super().check_finished(infer_request, response_choice, current_turn)

    def step(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
             current_turn: int) -> Dict:
        infer_request.messages.append({'role': 'user', 'content': self.tips_prompt})

        return {'infer_request': infer_request}

    def _is_thinking_template(self) -> bool:
        if not hasattr(self.infer_engine, 'template'):
            return False

        template = self.infer_engine.template
        return template.template_meta.is_thinking

    def _build_messages(self, original_messages: Messages) -> Messages:
        """
        Build history for a specific round, keeping only the think content from the last round.

        Args:
            original_messages: Original conversation messages

        Returns:
            Messages: History for this specific round
        """
        from copy import deepcopy

        # If this is a thinking template, use the template's method to prepare messages
        if self._is_thinking_template():
            # Create a mock inputs object to use the template's _swift_prepare_inputs method
            class MockInputs:

                def __init__(self, messages):
                    self.messages = deepcopy(messages)

            mock_inputs = MockInputs(original_messages)

            # Set up the template for inference mode
            template = self.infer_engine.template
            # _swift_prepare_inputs will remove historical thinking content when in train mode, patch the mode here
            original_mode = template.mode
            template.mode = 'train'
            # Use the template's method to prepare messages
            template._swift_prepare_inputs(mock_inputs)
            # Restore original mode
            template.mode = original_mode

            return mock_inputs.messages
        else:
            # Fallback to manual processing for non-thinking templates
            round_messages = []

            # Process messages in original order
            for i, msg in enumerate(original_messages):
                if msg['role'] == 'assistant' and isinstance(msg['content'], str) and i != len(original_messages) - 1:
                    # For assistant messages
                    assistant_no_think = msg['content'].split('</think>')[-1].strip()
                    round_messages.append(assistant_no_think)
                else:
                    round_messages.append(deepcopy(msg))

            return round_messages


class MathTipsScheduler(MultiTurnScheduler):
    tips_prompt = 'But wait... It seems I made a mistake,'

    def __init__(self, *args, **kwargs):
        from swift.rewards.orm import MathAccuracy
        super().__init__(*args, **kwargs)
        self.acc_func = kwargs.get('acc_function', MathAccuracy())
        # Cache the tokenized tips_prompt length for loss mask computation
        self._tips_token_ids = None

    def _get_tips_token_ids(self, tokenizer) -> List[int]:
        """Get tokenized tips_prompt (cached for efficiency)."""
        if self._tips_token_ids is None:
            # Tokenize without special tokens to get the raw token ids
            self._tips_token_ids = tokenizer.encode(self.tips_prompt, add_special_tokens=False)
        return self._tips_token_ids

    def check_finished(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
                       current_turn: int) -> bool:
        last_completion = infer_request.messages[-1]['content']
        # we only give tips once
        if self.tips_prompt in last_completion:
            return True
        solution = infer_request.data_dict['solution']

        acc = self.acc_func([last_completion], [solution])[0]
        if acc == 1:
            return True

        return super().check_finished(infer_request, response_choice, current_turn)

    def step(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
             current_turn: int) -> Dict:
        completion = response_choice.message.content
        response_token_ids = list(response_choice.token_ids) if response_choice.token_ids else []

        # Extract logprobs from response_choice before any truncation
        rollout_logprobs = self._extract_logprobs_from_choice(response_choice)

        # Truncate completion at <answer> or </think> tags
        truncate_idx = len(completion)
        if '<answer>' in completion:
            truncate_idx = min(truncate_idx, completion.index('<answer>'))
        if '</think>' in completion:
            truncate_idx = min(truncate_idx, completion.index('</think>'))

        if truncate_idx < len(completion):
            # Need to truncate token_ids and logprobs as well
            truncated_completion = completion[:truncate_idx]
            if response_token_ids and self.tokenizer is not None:
                # Find the token index corresponding to the truncation point
                # by decoding progressively until we reach or exceed the truncation point
                token_truncate_idx = len(response_token_ids)
                for i in range(1, len(response_token_ids) + 1):
                    decoded = self.tokenizer.decode(response_token_ids[:i], skip_special_tokens=False)
                    if len(decoded) >= truncate_idx:
                        token_truncate_idx = i
                        break
                response_token_ids = response_token_ids[:token_truncate_idx]
                # Truncate logprobs to match
                if rollout_logprobs:
                    rollout_logprobs = rollout_logprobs[:token_truncate_idx]
            completion = truncated_completion

        # Add tips_prompt
        completion += self.tips_prompt

        # Compute loss_mask for tips tokens
        # Note: rollout_logprobs should NOT include tips tokens because:
        # 1. Tips tokens have loss_mask=0, so their labels will be -100
        # 2. completion_mask = (labels != -100), so tips tokens won't be in completion_mask
        # 3. rollout_logprobs must align with completion_mask, not response_token_ids
        if response_token_ids and self.tokenizer is not None:
            tips_token_ids = self._get_tips_token_ids(self.tokenizer)
            # Loss mask: original tokens = 1, tips tokens = 0
            response_loss_mask = [1] * len(response_token_ids) + [0] * len(tips_token_ids)
            # Append tips token ids to response
            response_token_ids = response_token_ids + tips_token_ids
            # Do NOT extend rollout_logprobs for tips tokens - they are masked out in completion_mask
        else:
            response_loss_mask = []

        # Update messages
        if infer_request.messages[-1]['role'] == 'assistant':
            if not infer_request.messages[-1]['content']:
                # Multi-turn continuation: pop the dummy input we add in last turn
                infer_request.messages.pop(-1)
            infer_request.messages[-1]['content'] = completion
        else:
            infer_request.messages.append({'role': 'assistant', 'content': completion})

        result = {'infer_request': infer_request}
        if response_token_ids:
            result['response_token_ids'] = response_token_ids
            result['response_loss_mask'] = response_loss_mask
            if rollout_logprobs:
                result['rollout_logprobs'] = rollout_logprobs
        return result


class GYMScheduler(RolloutScheduler):

    def __init__(self, infer_engine: GRPOVllmEngine, max_turns: Optional[int] = None, **kwargs):
        super().__init__(infer_engine, max_turns, **kwargs)
        self.gym_env_name = kwargs.get('gym_env', None)
        self.context_manager_name = kwargs.get('context_manager', None)

    async def _create_env(self, env_config: Dict) -> Env:
        """Create environment instance from configuration."""
        env_name = env_config.get('name', self.gym_env_name)
        if env_name not in envs:
            raise ValueError(f"Environment '{env_name}' not found. Available: {list(envs.keys())}")
        return envs[env_name](env_config)

    async def _create_context_manager(self, ctx_config: Dict) -> ContextManager:
        """Create context manager from configuration."""
        ctx_name = ctx_config.get('name', self.context_manager_name)

        if not ctx_name:
            ctx_name = 'dummyContextManager'

        if ctx_name not in context_managers:
            raise ValueError(f"Context manager '{ctx_name}' not found. Available: {list(context_managers.keys())}")

        return context_managers[ctx_name](ctx_config)

    async def _close_env_async(self, env: Env):
        """Safely close environment with async support."""
        if env is None:
            return

        try:
            if hasattr(env, 'close') and asyncio.iscoroutinefunction(env.close):
                await env.close()
            elif hasattr(env, 'close'):
                env.close()
        except Exception as e:
            # Log the exception but don't raise it to avoid masking other errors
            import logging
            logging.warning(f'Error closing environment: {e}')

    async def run(self, infer_request: 'RolloutInferRequest', request_config: 'RequestConfig',
                  **kwargs) -> 'RolloutOutput':
        """
        Execute the gym environment-based rollout:
        1. Initialize environment and context manager
        2. Run multi-turn interactions between LLM and environment
        3. Collect trajectory information and rewards
        """
        # Extract configurations from request
        env_config = infer_request.data_dict.get('env_config', {})
        ctx_config = infer_request.data_dict.get('ctx_config', {})

        # Create environment and context manager
        env = None
        context_manager = None
        try:
            env = await self._create_env(env_config)
            context_manager = await self._create_context_manager(ctx_config)

            # Initialize environment
            observation, info, system_message = await env.reset(infer_request)

            # Build initial messages
            messages: Messages = []
            if system_message:
                messages.append({'role': 'system', 'content': system_message})
            messages.append({'role': 'user', 'content': observation})

            current_request = deepcopy(infer_request)
            current_turn = 1
            done = False
            total_reward = 0.0
            step_rewards = []
            trajectory_id = infer_request.uuid
            trajectory_info = [info]

            while not done and current_turn <= (self.max_turns or float('inf')):
                # Apply context management (e.g., history compression)
                messages = context_manager.manage_context(messages, trajectory_id)
                current_request.messages = messages
                remove_response(current_request.messages)

                response: 'ChatCompletionResponse' = await self.infer_engine.infer_async(
                    current_request, request_config, **kwargs)
                response_choice: 'ChatCompletionResponseChoice' = response.choices[0]
                completion = response_choice.message.content
                messages.append({'role': 'assistant', 'content': completion})

                # Execute environment step
                next_obs, reward, done, step_info = await env.step(deepcopy(messages))

                # Update trajectory information
                total_reward += reward
                step_rewards.append(reward)
                trajectory_info.append(step_info)

                # Prepare for next turn
                if not done:
                    messages.append({'role': 'user', 'content': next_obs})
                    current_request.messages = messages
                    current_turn += 1

            return RolloutOutput(
                response=response,
                messages=messages,
                rollout_infos={
                    'num_turns': current_turn,
                    'trajectory_id': trajectory_id,
                    'total_reward': total_reward,
                    'step_rewards': step_rewards,
                    'trajectory_info': trajectory_info
                })

        finally:
            # Ensure environment is properly closed
            if env is not None:
                await self._close_env_async(env)

import concurrent.futures
import inspect

class IterResearchScheduler(MultiTurnScheduler):
    """
    专为 IterResearch (Think-Report-Action) 架构设计的强化学习多轮调度器。

    采用完全定制的 run() 覆盖，但在内部回归 check_finished() 和 step() 的标准生命周期。
    """

    def __init__(self, *args, **kwargs):
        print(">>> [DEBUG] 成功进入 IterResearchScheduler.__init__", flush=True)
        super().__init__(*args, **kwargs)

        base_dir = "/root"
        mcp_config_path = os.path.join(base_dir, "RL", "tools", "mcp_configs.json")
        cache_path = os.path.join(base_dir, "mcp_tools_cache.json")
        
        print(">>> [DEBUG] 开始加载工具", flush=True)

        try:
            tool_config = ToolConfig(mcp_config_path=mcp_config_path, cache_path=cache_path)
            selected_tool_indices = [0, 1, 2, 3]
            # --- 核心改造：使用新线程执行异步任务 ---
            def _sync_wrapper():
                # 这是一个全新的线程，没有运行中的 event loop，所以可以安全调用 asyncio.run
                return asyncio.run(tool_config.get_tools_by_indices(selected_tool_indices))
            
            # 开启一个包含1个工作线程的线程池
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # 提交任务，并调用 .result() 阻塞当前的 __init__，直到异步加载全部完成
                tools = executor.submit(_sync_wrapper).result()
            # ---------------------------------------
        except Exception as e:
            # 建议将 debug_logger 替换为标准的 logging，或确保其已在全局初始化
            print(f"工具加载失败: {e}", flush=True)
            tools = []

        self.tool_map = {tool.name: tool for tool in tools}
        self.max_steps = self.max_turns or 100
        self.parser = LLMOutputParser()

        self.citation_prompt = """你是一个深度研究智能体。你需要通过“思考→报告→行动”(Think-Report-Action) 的迭代循环来解决复杂问题。
你没有历史对话记录！你唯一的知识载体和上下文是“全局演进报告 (report)”。你必须在每一次迭代中，将新线索与旧报告进行深度语义融合、去重和提炼，而非简单堆砌。

**输出结构要求 (必须且只能包含以下三个 XML 标签，禁止输出任何标签外的文本)**：
1. <think> 认知暂存区：分析当前工作区状态，评估上次行动的结果(observation)，准备更新报告 (report)并推理下一步工具调用策略(action)。此部分属于你的内部推理，不会带入下一轮。</think>
2. <report> 演进报告 (中心记忆)：这是你唯一的上下文记忆！请将本次新发现与上一轮的旧报告进行综合，输出一份结构化、高信息密度的最新进展摘要。禁止简单追加，必须进行语义融合。</report>
3. <action> 行动：仅使用提供的工具，格式为"工具名[参数]"。如果认为已有足够证据回答初始问题，请输出"finish[最终回复]"，这是你向用户展示结果的唯一渠道。</action>

**信息溯源与引用规范**
1. 你的所有事实陈述必须有据可查。
2. 在更新 <report> 时，必须在报告的最下方维护一个 [References] 列表。格式为：\n[1] 标题 (URL或来源ID)。
3. 在最终通过 finish[] 提交答案时，必须使用内联引用格式（如“该算法在多步推理上表现优异 [1][3]”），并在finish[]中附上完整的参考文献列表。

支持工具：
{tool_descriptions}

原始研究任务 (Question)：{task}

【当前工作区状态】
当前时间：{current_datetime}
上一轮报告 (Report_{i-1})：
{report}

上一轮行动 (Action_{i-1})：{prev_action}
上一轮结果 (Observation_{i-1})：{prev_observation}

请输出当前步骤的 <think>, <report> 和 <action> (仅输出XML标签)：
"""
        print(">>> [DEBUG] __init__ 结束", flush=True)

    def _build_prompt(self, task: str, workspace_state: dict) -> str:
        tool_descriptions = "\n".join([
            f"- {name}：{tool.description}"
            for name, tool in self.tool_map.items()
        ])
        return self.citation_prompt.format(
            tool_descriptions=tool_descriptions,
            task=task,
            current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            report=workspace_state["report"],
            prev_action=workspace_state["prev_action"],
            prev_observation=workspace_state["prev_observation"],
            i="{i}"  # 规避 format 解析大括号时的冲突，如果需要动态回合数可以在这里注入
        ).strip()

    async def _execute_action(self, action: str) -> str:
        if action.startswith("finish["):
            return "任务完成"
        try:
            tool_name, params = self.parser.parse_tool_action(action, list(self.tool_map.keys()))
            if tool_name in self.tool_map:
                tool = self.tool_map[tool_name]
                # 🌟 修复 4：动态判断工具的 run 方法是同步还是异步
                if inspect.iscoroutinefunction(tool.run):
                    return await tool.run(params)  # 异步工具（如 MCP），用 await
                else:
                    return tool.run(params)  # 同步工具（如本地算个加减法），直接调
            else:
                return f"工具不存在：{tool_name}"
        except ValueError:
            return f"无效行动：{action}，支持的工具为{list(self.tool_map.keys())}"

    async def run(self, infer_request: 'RolloutInferRequest', request_config: 'RequestConfig',
                  **kwargs) -> List['RolloutOutput']:
        # 初始提问作为全局任务
        task = infer_request.messages[0]['content']
        current_request = infer_request
        current_turn = 1
        rollout_outputs = []

        # 将工作区初始化放在 run 内部是正确的，这保证了并发 batch 请求之间状态隔离
        workspace = StateManager()

        # 第一轮：初始化无记忆 Prompt
        current_state = workspace.get_current_state()
        current_request.messages = [{'role': 'user', 'content': self._build_prompt(task, current_state)}]

        while True:
            print(f"========== 开始第 {current_turn} 轮迭代 ==========", flush=True)

            # 1. 异步推理获取模型输出
            response: 'ChatCompletionResponse' = await self.infer_engine.infer_async(current_request, request_config, **kwargs)
            response_choice: 'ChatCompletionResponseChoice' = response.choices[0]
            completion = response_choice.message.content

            # 关键修复：使用 copy() 隔离每轮的 messages，避免下一轮直接覆盖导致历史轨迹被污染
            current_messages = current_request.messages.copy()
            current_messages.append({'role': 'assistant', 'content': completion})

            # 2. 封装并保存当前独立轮次的 Rollout 轨迹
            round_output = RolloutOutput(
                response=response,
                messages=current_messages,
                response_token_ids=response_choice.token_ids,
                rollout_infos={'num_turns': current_turn, 'task': task}
            )
            rollout_outputs.append(round_output)

            # 3. 解析模型输出，与环境（工具）交互
            try:
                think, report, action = self.parser.parse_tra(completion)
                if not report:
                    report = workspace.get_current_state().get("report", "")

                observation = self._execute_action(action)
                workspace.update_state(think, report, action, observation)
                parsed_action = action
            except Exception as e:
                parsed_action = "parse_error"
                observation = f"解析错误或格式不合规：{e}"
                workspace.update_state("解析失败", workspace.get_current_state().get("report", ""), parsed_action,
                                       observation)

            # 4. 判断终止条件
            if self.check_finished(infer_request, response_choice, current_turn, parsed_action=parsed_action):
                break

            # 5. 状态机流转：准备下一轮 Request
            ret = self.step(current_request, response_choice, current_turn, task=task,
                            current_state=workspace.get_current_state())
            current_request = ret['infer_request']

            current_turn += 1

        return rollout_outputs

    def check_finished(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
                       current_turn: int, parsed_action: str = None) -> bool:
        """
        判断 Rollout 是否结束。
        保留了原本 check_finished 的函数签名以兼容基类设计。
        """
        if parsed_action and parsed_action.startswith("finish["):
            return True
        if self.max_steps and current_turn >= self.max_steps:
            return True
        # 兜底：处理因为长度截断等异常结束的场景
        if response_choice.finish_reason == 'length':
            return True
        return False

    def step(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
             current_turn: int, task: str = None, current_state: dict = None) -> Dict:
        """
        接管下一轮 Prompt 的生成。
        彻底抛弃传统多轮拼接，用工作区的最新状态重写 infer_request.messages。
        """
        new_prompt = self._build_prompt(task, current_state)
        # 直接覆盖 messages 数组，强制模型只看最新的一轮 report
        infer_request.messages = [{'role': 'user', 'content': new_prompt}]
        return {'infer_request': infer_request}

multi_turns = {
    'math_tip_trick': MathTipsScheduler,
    'gym_scheduler': GYMScheduler,
    'iter_research_scheduler': IterResearchScheduler,
    'thinking_tips_scheduler': ThinkingModelTipsScheduler,
}
