import itertools
import torch
from torch.autograd import DeviceType

from collections import defaultdict, namedtuple
from operator import attrgetter

from typing import Dict, List, Tuple, Optional

import bisect
import math

import torch.autograd.profiler_util
import torch.autograd.profiler


def _build_table(
        events,
        sort_by=None,
        header=None,
        row_limit=100,
        max_src_column_width=75,
        with_flops=False,
        profile_memory=False,
        top_level_events_only=False):
    """Prints a summary of events (which can be a list of FunctionEvent or FunctionEventAvg)."""
    print("in monkeypatch")
    if len(events) == 0:
        return ""

    has_cuda_time = any([event.self_cuda_time_total > 0 for event in events])
    has_cuda_mem = any([event.self_cuda_memory_usage > 0 for event in events])
    has_input_shapes = any(
        [(event.input_shapes is not None and len(event.input_shapes) > 0) for event in events])

    if sort_by is not None:
        events = pfu.EventList(sorted(
            events, key=lambda evt: getattr(evt, sort_by), reverse=True
        ), use_cuda=has_cuda_time, profile_memory=profile_memory, with_flops=with_flops)

    MAX_NAME_COLUMN_WIDTH = 150
    name_column_width = max([len(evt.key) for evt in events]) + 4
    name_column_width = min(name_column_width, MAX_NAME_COLUMN_WIDTH)

    MAX_SHAPES_COLUMN_WIDTH = 80
    shapes_column_width = max([len(str(evt.input_shapes)) for evt in events]) + 4
    shapes_column_width = min(shapes_column_width, MAX_SHAPES_COLUMN_WIDTH)

    DEFAULT_COLUMN_WIDTH = 12
    flops_column_width = DEFAULT_COLUMN_WIDTH

    src_column_width = None
    stacks = []
    for evt in events:
        if evt.stack is not None and len(evt.stack) > 0:
            stacks.append(evt.stack)
    has_stack = len(stacks) > 0
    if has_stack:
        src_column_width = max([max([len(entry) for entry in stack]) for stack in stacks]) + 4
        src_column_width = min(src_column_width, max_src_column_width)

    headers = [
        'Name',
        'Self CPU %',
        'Self CPU',
        'CPU total %',
        'CPU total',
        'CPU time avg',
    ]
    if has_cuda_time:
        headers.extend([
            'Self CUDA',
            'Self CUDA %',
            'CUDA total',
            'CUDA time avg',
        ])
    if profile_memory:
        headers.extend([
            'CPU Mem',
            'Self CPU Mem',
        ])
        if has_cuda_mem:
            headers.extend([
                'CUDA Mem',
                'Self CUDA Mem',
            ])
    headers.append(
        '# of Calls'
    )
    # Only append Node ID if any event has a valid (>= 0) Node ID
    append_node_id = any([evt.node_id != -1 for evt in events])
    if append_node_id:
        headers.append('Node ID')

    # Have to use a list because nonlocal is Py3 only...
    SPACING_SIZE = 2
    row_format_lst = [""]
    header_sep_lst = [""]
    line_length_lst = [-SPACING_SIZE]
    MAX_STACK_ENTRY = 5

    def add_column(padding, text_dir='>'):
        row_format_lst[0] += '{: ' + text_dir + str(padding) + '}' + (' ' * SPACING_SIZE)
        header_sep_lst[0] += '-' * padding + (' ' * SPACING_SIZE)
        line_length_lst[0] += padding + SPACING_SIZE

    def auto_scale_flops(flops):
        flop_headers = [
            'FLOPs',
            'KFLOPs',
            'MFLOPs',
            'GFLOPs',
            'TFLOPs',
            'PFLOPs',
        ]
        assert flops > 0
        log_flops = max(0, min(math.log10(flops) / 3, float(len(flop_headers) - 1)))
        assert log_flops >= 0 and log_flops < len(flop_headers)
        return (pow(10, (math.floor(log_flops) * -3.0)), flop_headers[int(log_flops)])

    add_column(name_column_width)
    for _ in headers[1:]:
        add_column(DEFAULT_COLUMN_WIDTH)

    if has_input_shapes:
        headers.append('Input Shapes')
        add_column(shapes_column_width)

    if has_stack:
        headers.append('Source Location')
        add_column(src_column_width, text_dir='<')

    if with_flops:
        # Auto-scaling of flops header
        raw_flops = []
        for evt in events:
            if evt.flops > 0:
                raw_flops.append(evt.flops)
        if len(raw_flops) != 0:
            (flops_scale, flops_header) = auto_scale_flops(min(raw_flops))
            headers.append('Total {}'.format(flops_header))
            add_column(flops_column_width)
        else:
            with_flops = False  # can't find any valid flops

    row_format = row_format_lst[0]
    header_sep = header_sep_lst[0]
    line_length = line_length_lst[0]
    add_column = None  # type: ignore[assignment]

    # Have to use a list because nonlocal is Py3 only...
    result = []

    def append(s):
        result.append(s)
        result.append('\n')  # Yes, newline after the end as well

    sum_self_cpu_time_total = sum([event.self_cpu_time_total for event in events])
    sum_self_cuda_time_total = 0
    for evt in events:
        if evt.device_type == DeviceType.CPU:
            # in legacy profiler, kernel info is stored in cpu events
            if evt.is_legacy:
                sum_self_cuda_time_total += evt.self_cuda_time_total
        elif evt.device_type == DeviceType.CUDA:
            # in kineto profiler, there're events with the correct device type (e.g. CUDA)
            sum_self_cuda_time_total += evt.self_cuda_time_total

    # Actual printing
    if header is not None:
        append('=' * line_length)
        append(header)
    if top_level_events_only:
        append('=' * line_length)
        append('This report only display top-level ops statistics')
    append(header_sep)
    append(row_format.format(*headers))

    append(header_sep)

    def trim_path(path, src_column_width):
        if len(path) > src_column_width:
            offset = len(path) - src_column_width
            path = path[offset:]
            if len(path) > 3:
                path = "..." + path[3:]
        return path

    event_limit = 0
    for evt in events:
        if event_limit == row_limit:
            break
        if top_level_events_only and evt.cpu_parent is not None:
            continue
        else:
            event_limit += 1
        name = evt.key
        if len(name) >= MAX_NAME_COLUMN_WIDTH - 3:
            name = name[:(MAX_NAME_COLUMN_WIDTH - 3)] + "..."
        row_values = [
            name,
            # Self CPU total %, 0 for async events.
            _format_time_share(evt.self_cpu_time_total, sum_self_cpu_time_total),
            evt.self_cpu_time_total_str,  # Self CPU total
            # CPU total %, 0 for async events.
            _format_time_share(evt.cpu_time_total, sum_self_cpu_time_total) if not evt.is_async else 0,
            evt.cpu_time_total_str,  # CPU total
            evt.cpu_time_str,  # CPU time avg
        ]
        if has_cuda_time:
            row_values.extend([
                evt.self_cuda_time_total_str,
                # CUDA time total %
                _format_time_share(evt.self_cuda_time_total, sum_self_cuda_time_total),
                evt.cuda_time_total_str,
                evt.cuda_time_str,  # Cuda time avg
            ])
        if profile_memory:
            row_values.extend([
                # CPU Mem Total
                _format_memory(evt.cpu_memory_usage),
                # Self CPU Mem Total
                _format_memory(evt.self_cpu_memory_usage),
            ])
            if has_cuda_mem:
                row_values.extend([
                    # CUDA Mem Total
                    _format_memory(evt.cuda_memory_usage),
                    # Self CUDA Mem Total
                    _format_memory(evt.self_cuda_memory_usage),
                ])
        row_values.append(
            evt.count,  # Number of calls
        )

        if append_node_id:
            row_values.append(evt.node_id)
        if has_input_shapes:
            row_values.append(str(evt.input_shapes)[:shapes_column_width])
        if with_flops:
            if evt.flops <= 0:
                row_values.append("--")
            else:
                row_values.append('{0:8.3f}'.format(evt.flops * flops_scale))
        if has_stack:
            src_field = ""
            if len(evt.stack) > 0:
                src_field = trim_path(evt.stack[0], src_column_width)
            row_values.append(src_field)
        append(row_format.format(*row_values))

        if has_stack:
            empty_headers = [""] * (len(headers) - 1)
            for entry in evt.stack[1:MAX_STACK_ENTRY]:
                append(row_format.format(*(empty_headers + [trim_path(entry, src_column_width)])))
            empty_headers.append("")
            append(row_format.format(*empty_headers))

    append(header_sep)
    append("Self CPU time total: {}".format(_format_time(sum_self_cpu_time_total)))
    if has_cuda_time:
        append("Self CUDA time total: {}".format(_format_time(sum_self_cuda_time_total)))
    return ''.join(result)


torch.autograd.profiler_util.EventList._build_table = _build_table
torch.autograd.profiler.EventList = torch.autograd.profiler_util.EventList


def key_averages(self, group_by_input_shape: bool = False, group_by_stack_n: int = 0):
    print('in monkeypatch key_averages')
    self._check_finish()
    assert self.function_events is not None, "Expected profiling results"
    return self.function_events.key_averages(group_by_input_shape, group_by_stack_n)

torch.autograd.profiler.profile.key_averages = key_averages
