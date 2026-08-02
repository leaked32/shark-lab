/*
 * Project: shark-lab
 * Repository: https://github.com/leaked32/shark-lab
 *
 * File: chatty/discrete.cpp
 *
 * License: MIT
 */

#include "app.hpp"

#include "imgui.h"
#include <cstring>
#include <string>
#include <vector>

bool show_lorebook_window = false;

static char lore_keyword[256] = "";
static char lore_content[2048] = "";
static int selected_lore_index = -1;

struct lore_entry_ui
{
	int id;
	std::string keyword;
	std::string content;
};

static std::vector<lore_entry_ui> lore_entries;

void RenderLorebookWindow(
	ApplicationState& app_state, uint32_t selected_peer_id)
{
	if (!show_lorebook_window)
		return;

	if (!app_state.uni_db_->get_peer_by_id(selected_peer_id).has_value()) {
		ImGui::OpenPopup("Lorebook");
		if (ImGui::BeginPopupModal("Lorebook", &show_lorebook_window)) {
			ImGui::Text("No peer selected.");
			ImGui::EndPopup();
		}
		return;
	}

	// Ensure popup is opened once
	ImGui::OpenPopup("Lorebook");

	ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);

	if (!ImGui::BeginPopupModal("Lorebook", &show_lorebook_window))
		return;

	// -----------------------------
	// Load current lore entries
	// -----------------------------
	lore_entries.clear();
	auto rows = app_state.uni_db_->get_lorebook_for_peer(selected_peer_id);

	for (auto& r : rows)
		lore_entries.push_back({(int)r.id, r.keyword, r.content});

	// -----------------------------
	// LEFT: list
	// -----------------------------
	ImGui::BeginChild("##lore_list", ImVec2(200, 0), true);

	for (int i = 0; i < (int)lore_entries.size(); i++) {
		bool selected = (selected_lore_index == i);

		if (ImGui::Selectable(lore_entries[i].keyword.c_str(), selected)) {
			selected_lore_index = i;

			strncpy(lore_keyword, lore_entries[i].keyword.c_str(), sizeof(lore_keyword));

			strncpy(lore_content, lore_entries[i].content.c_str(), sizeof(lore_content));
		}
	}

	ImGui::EndChild();
	ImGui::SameLine();

	// -----------------------------
	// RIGHT: editor
	// -----------------------------
	ImGui::BeginChild("##lore_editor", ImVec2(0, 0), false);

	ImGui::Text("Keyword:");
	ImGui::InputText("##keyword", lore_keyword, sizeof(lore_keyword));

	ImGui::Text("Content:");

	ImVec2 avail = ImGui::GetContentRegionAvail();

	float button_height = ImGui::GetFrameHeight() + ImGui::GetStyle().ItemSpacing.y;

	ImGui::InputTextMultiline("##content", lore_content, sizeof(lore_content),
							  ImVec2(avail.x, avail.y - button_height));

	// ImGui::Spacing();

	if (ImGui::Button("Add / Update")) {
		if (selected_lore_index >= 0) {
			app_state.uni_db_->update_lorebook_entry(lore_entries[selected_lore_index].id,
													 lore_keyword, lore_content);
		}
		else {
			app_state.uni_db_->insert_lorebook(selected_peer_id, lore_keyword, lore_content);
		}

		lore_keyword[0] = 0;
		lore_content[0] = 0;
		selected_lore_index = -1;
	}

	ImGui::SameLine();

	if (ImGui::Button("Delete") && selected_lore_index >= 0) {
		app_state.uni_db_->delete_lorebook_entry(lore_entries[selected_lore_index].id);

		selected_lore_index = -1;
		lore_keyword[0] = 0;
		lore_content[0] = 0;
	}

	ImGui::EndChild();
	ImGui::EndPopup();
}

void RenderModalText(
	ApplicationState& app_state, ActivityModalText& modal_text)
{
	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
	// const char* title, std::unique_ptr<std::string>& message
	if (ImGui::Begin(modal_text.title_.c_str(), &modal_text.open, ImGuiWindowFlags_None)) {
		// Make a temporary buffer

		ImGui::TextUnformatted("Message:");

		ImVec2 avail = ImGui::GetContentRegionAvail();

		float button_height = ImGui::GetFrameHeight() + ImGui::GetStyle().ItemSpacing.y;

		ImGui::BeginChild("Message", ImVec2(avail.x, avail.y - button_height));

		ImGui::PushTextWrapPos();
		ImGui::TextUnformatted(modal_text.text_.c_str());
		ImGui::PopTextWrapPos();

		ImGui::EndChild();
		/*
		 *		ImGui::InputTextMultiline(
		 *			"##message",
		 *			message->data(), message->size() + 1, ImVec2(600, 450),
		 *				 ImGuiInputTextFlags_ReadOnly);
		 */
		if (ImGui::Button("Copy"))
			ImGui::SetClipboardText(modal_text.text_.c_str());

		ImGui::SameLine();

		if (ImGui::Button("OK")) {
			modal_text.open = false;
			// ImGui::CloseCurrentPopup();
		}

		ImGui::End();
	}
}

void RenderPeerEditorWindow(
	ApplicationState& app_state, ActivityPeerEditor& peer_state)
{
	if (!peer_state.open)
		return;

	if (!ImGui::IsPopupOpen("Peer Editor"))
		ImGui::OpenPopup("Peer Editor");

	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);

	/*
	ImGui::Begin(
		"Login",
		nullptr,
		ImGuiWindowFlags_NoResize |
		ImGuiWindowFlags_NoMove |
		ImGuiWindowFlags_NoCollapse
	);
	*/

	if (ImGui::Begin("Peer Editor", &peer_state.open,
					 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
						 ImGuiWindowFlags_NoCollapse))
	{
		ImGui::Text("Name:");
		ImGui::InputText("##name", peer_state.name, sizeof(peer_state.name));

		ImGui::Separator();
		ImGui::Text("Character card:");

		float reserved = ImGui::GetFrameHeight() + ImGui::GetStyle().ItemSpacing.y * 1;
		// Scrollable log
		ImGui::InputTextMultiline("##card", peer_state.card, sizeof(peer_state.card),
								  ImVec2(-1, -reserved),
								  ImGuiInputTextFlags_AllowTabInput | ImGuiInputTextFlags_WordWrap);

		if (ImGui::Button(peer_state.mode == ActivityPeerEditor::Mode::create ? "Create" : "Save"))
		{
			std::string name(peer_state.name);
			std::string card(peer_state.card);

			if (!name.empty()) {
				if (peer_state.mode == ActivityPeerEditor::Mode::create) {
					app_state.uni_db_->insert_peer(name, card);
				}
				else {
					app_state.uni_db_->update_peer(peer_state.selected_peer_id_, name, card);
				}

				peer_state.name[0] = 0;
				peer_state.card[0] = 0;
				peer_state.selected_peer_id_ = 0;
				peer_state.open = false;

				// ImGui::CloseCurrentPopup();
			}
		}

		ImGui::SameLine();

		if (ImGui::Button("Cancel")) {
			peer_state.name[0] = 0;
			peer_state.card[0] = 0;
			peer_state.selected_peer_id_ = 0;
			peer_state.open = false;

			ImGui::CloseCurrentPopup();
		}

		ImGui::End();
	}
}
