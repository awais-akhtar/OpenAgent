import unittest
from tempfile import TemporaryDirectory
from openagent.storage import OpenAgentStore
from openagent.personal_os import PersonalOS, PersonalOSError
from openagent.vault import EncryptedVault

class RemainingLocalGapTests(unittest.TestCase):
    def test_personal_records_update_and_delete_are_scoped(self):
        with TemporaryDirectory() as tmp:
            personal=PersonalOS(OpenAgentStore(tmp))
            note=personal.save_note(title="First",body="Original",project="p",owner="alice")
            draft=personal.save_draft(title="First",content="Original",project="p",owner="alice")
            signature=personal.upsert_signature(label="First",content="Original",project="p",owner="alice")
            for project,owner in [("p","bob"),("forged","alice")]:
                with self.assertRaises(PersonalOSError):
                    personal.save_note(note_id=note["id"],title="Forged",project=project,owner=owner)
                with self.assertRaises(PersonalOSError):
                    personal.save_draft(draft_id=draft["id"],title="Forged",content="Forged",project=project,owner=owner)
                with self.assertRaises(PersonalOSError):
                    personal.upsert_signature(signature_id=signature["id"],label="Forged",content="Forged",project=project,owner=owner)
                for collection,record in [("notes",note),("drafts",draft),("signatures",signature)]:
                    self.assertFalse(personal.delete_record(collection,record["id"],project=project,owner=owner))
            changed=personal.save_note(note_id=note["id"],title="Renamed",body="New",project="p",owner="alice")
            self.assertEqual(changed["id"],note["id"])
            changed_draft=personal.save_draft(draft_id=draft["id"],title="Renamed",content="New",project="p",owner="alice")
            self.assertEqual(changed_draft["version"],2)
            self.assertEqual(len(personal.draft_versions(draft["id"])),2)
            changed_signature=personal.upsert_signature(signature_id=signature["id"],label="Renamed",content="New",project="p",owner="alice")
            self.assertEqual(changed_signature["id"],signature["id"])
            self.assertEqual(changed_signature["label"],"Renamed")
            for collection,record in [("notes",note),("drafts",draft),("signatures",signature)]:
                self.assertTrue(personal.delete_record(collection,record["id"],project="p",owner="alice"))
            restored=PersonalOS(OpenAgentStore(tmp))
            self.assertEqual(restored.notes(project="p",owner="alice"),[])
            self.assertEqual(restored.drafts(project="p",owner="alice"),[])
            self.assertEqual(restored.signatures(project="p",owner="alice"),[])
            self.assertEqual(restored.draft_versions(draft["id"]),[])

    def test_concurrent_draft_edits_preserve_both_versions(self):
        from concurrent.futures import ThreadPoolExecutor
        with TemporaryDirectory() as tmp:
            service=PersonalOS(OpenAgentStore(tmp))
            draft=service.save_draft(title="Concurrent",content="v1",project="p",owner="alice")
            def edit(content):
                service.save_draft(draft_id=draft["id"],title="Concurrent",content=content,project="p",owner="alice")
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(edit,["second edit","third edit"]))
            versions=service.draft_versions(draft["id"])
            self.assertEqual([version["version"] for version in versions],[3,2,1])
            self.assertEqual({version["content"] for version in versions},{"v1","second edit","third edit"})

    def test_vault_edit_delete_only_pending_in_matching_project(self):
        with TemporaryDirectory() as tmp:
            vault=EncryptedVault(OpenAgentStore(tmp))
            entry=vault.store_secret("Synthetic", "noncredential", "example-test-only", project="p")
            request=vault.create_access_request("Original",[entry["id"]],project="p")
            with self.assertRaises(KeyError): vault.delete_access_request(request["id"],project="other")
            with self.assertRaises(ValueError):
                vault.update_access_request(request["id"],reason="Changed",requested_entries=["missing"],project="p")
            changed=vault.update_access_request(request["id"],reason="Changed",requested_entries=[entry["id"]],duration_minutes=15,project="p")
            self.assertEqual(changed["created_at"],request["created_at"])
            self.assertEqual(changed["reason"],"Changed")
            self.assertTrue(vault.delete_access_request(request["id"],project="p")["deleted"])
            reviewed=vault.create_access_request("Reviewed",[entry["id"]],project="p")
            vault.approve_access_request(reviewed["id"],project="p")
            with self.assertRaises(ValueError): vault.delete_access_request(reviewed["id"],project="p")
            with self.assertRaises(ValueError):
                vault.update_access_request(reviewed["id"],reason="Changed",requested_entries=[entry["id"]],project="p")
            self.assertEqual(EncryptedVault(OpenAgentStore(tmp)).list_access_requests(project="p")[0]["status"],"approved")
