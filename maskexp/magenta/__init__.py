# Copyright 2023 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Pulls in all magenta libraries that are in the public API.."""

import maskexp.magenta.common.beam_search
import maskexp.magenta.common.concurrency
import maskexp.magenta.common.nade
import maskexp.magenta.common.sequence_example_lib
import maskexp.magenta.common.state_util
import maskexp.magenta.common.testing_lib
import maskexp.magenta.common.tf_utils
import maskexp.magenta.pipelines.dag_pipeline
import maskexp.magenta.pipelines.drum_pipelines
import maskexp.magenta.pipelines.lead_sheet_pipelines
import maskexp.magenta.pipelines.melody_pipelines
import maskexp.magenta.pipelines.note_sequence_pipelines
import maskexp.magenta.pipelines.pipeline
import maskexp.magenta.pipelines.pipelines_common
import maskexp.magenta.pipelines.statistics
import maskexp.magenta.version
from maskexp.magenta.version import __version__
