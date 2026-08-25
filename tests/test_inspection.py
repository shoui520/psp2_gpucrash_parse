import unittest

from psp2_gpucrash_parse.debugger import load_debug_state, thread_detail_report
from psp2_gpucrash_parse.inspection import (
    annotated_stack,
    classify_value,
    modules_report,
    retained_map,
    xref_report,
)
from psp2_gpucrash_parse.registers import RegisterDatabase
from psp2_gpucrash_parse.unwind import unwind_thread
from psp2_gpucrash_parse.validation import validate_core
from psp2_gpucrash_parse.waitgraph import wait_graph
from tests.test_debugger import dormant_execution_core, execution_core


class InspectionTests(unittest.TestCase):
    def test_map_and_module_views_combine_metadata_with_capture_ranges(self):
        core = execution_core()
        address_map = retained_map(core)
        kinds = {item["kind"] for item in address_map["entries"]}
        self.assertIn("module-segment", kinds)
        self.assertIn("thread-stack", kinds)
        self.assertIn("load-segment", kinds)
        module = modules_report(core)["modules"][0]
        self.assertEqual(module["name"], "TestModule")
        self.assertEqual(module["execution_threads"][0]["name"], "RenderTask")

    def test_stack_words_and_xrefs_are_classified_by_source(self):
        core = execution_core()
        database = RegisterDatabase()
        report, diagnosis = thread_detail_report(core, database, "RenderTask", 16)
        stack = annotated_stack(core, diagnosis, report["thread"], 16)
        self.assertEqual(stack["words"][0]["value"], 0x81000081)
        self.assertEqual(
            stack["words"][0]["classifications"][0]["kind"],
            "module-address",
        )
        xrefs = xref_report(core, database, 0x81000081)
        self.assertEqual(xrefs["memory_reference_count"], 1)
        self.assertEqual(
            xrefs["memory_references"][0]["source_classification"],
            "thread-stack",
        )

    def test_dormant_context_does_not_seed_stack_annotation_or_unwind(self):
        core = dormant_execution_core()
        database = RegisterDatabase()
        report, diagnosis = thread_detail_report(core, database, "RenderTask", 16)
        thread = report["thread"]
        stack = annotated_stack(core, diagnosis, thread, 16)
        self.assertEqual(stack["status"], "non-live-thread-context")
        self.assertEqual(stack["words"], [])
        module = modules_report(core)["modules"][0]
        self.assertEqual(module["execution_threads"], [])
        self.assertEqual(len(module["threads_with_retained_pc"]), 1)
        unwind = unwind_thread(core, diagnosis, thread, [], 4)
        self.assertEqual(unwind["status"], "non-live-thread-context")
        self.assertEqual(unwind["frames"], [])

    def test_null_gpu_fault_page_does_not_classify_small_scalars_as_pointers(self):
        indexes = {
            "modules": None,
            "blocks": [],
            "objects": {},
            "threads_by_uid": {},
            "threads": [],
            "resources": [],
            "faults": [
                {
                    "index": 0,
                    "gpu_virtual_address": 0,
                    "requestor_names": ["test"],
                }
            ],
        }
        self.assertEqual(classify_value(execution_core(), indexes, 3), [])
        self.assertEqual(classify_value(execution_core(), indexes, 0x10), [])

    def test_wait_graph_and_validator_report_only_retained_evidence(self):
        core = execution_core()
        _info, diagnosis = load_debug_state(core, RegisterDatabase())
        graph = wait_graph(core, RegisterDatabase(), diagnosis)
        self.assertEqual(graph["cycle_count"], 0)
        self.assertIn("captured", graph["qualification"])
        validation = validate_core(core)
        self.assertFalse(validation["valid"])
        failed = {item["code"] for item in validation["checks"] if not item["passed"]}
        self.assertIn("fixed-note-set", failed)


if __name__ == "__main__":
    unittest.main()
