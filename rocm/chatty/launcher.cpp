/*
 * Project: shark-lab
 * Repository: https://github.com/leaked32/shark-lab
 *
 * File: chatty/launcher.cpp
 *
 * License: MIT
 */

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "imgui.h"
#include <boost/json.hpp>
#include <GLFW/glfw3.h>

// #include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
// #include <ranges>
#include "app.hpp"
#include "external.hpp"

void RenderLoginPanel(
	ApplicationState& app_state,
	ActivityState& act_state
)
{
	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
	
	ImGui::Begin(
		"Login",
		nullptr,
		ImGuiWindowFlags_NoResize |
		ImGuiWindowFlags_NoMove |
		ImGuiWindowFlags_NoCollapse
	);
	auto peers = app_state.uni_db_->get_all_peers();
	static int selected = -1;
	float left_width = 300.0f;
	
	ImGui::BeginChild("PeerList", ImVec2(left_width, 0),true);
	
	ImGui::Text("Select identity:");
	ImGui::Separator();
	
	if (ImGui::BeginListBox(
		"##peers",
		ImVec2(-FLT_MIN, -FLT_MIN)
	)) {
		for (int i = 0; i < peers.size(); i++) {
			bool active = selected == i;
			
			std::string label = peers[i].name + "##" + std::to_string(peers[i].id);
			
			if (ImGui::Selectable(label.c_str(), active)) {
				selected = i;
			}
			
			if (active) {
				ImGui::SetItemDefaultFocus();
			}
		}
		
		ImGui::EndListBox();
	}
	ImGui::EndChild();
	ImGui::SameLine();
	
	// Right: preview
	ImGui::BeginChild("PeerPreview", ImVec2(0, 0), true);
	
	if (selected >= 0 && selected < peers.size()) {
		auto& p = peers[selected];
		ImGui::Text("Name: %s", p.name.c_str());
		
		ImGui::Separator();
		if (!p.card.empty()) {
			ImGui::TextWrapped("%s", p.card.c_str());
		} else {
			ImGui::TextDisabled("No card");
		}
		
		ImGui::Spacing();
		if (ImGui::Button("Login",ImVec2(120, 40))) {
			app_state.self_id_ = p.id;
			act_state.open = false;
			app_state.states_.emplace_back(
				std::make_unique<ActivityChat>()
			);
		}
		
	} else {
		ImGui::TextDisabled("Select a peer");
	}
	
	ImGui::EndChild();
	ImGui::End();
}

void chat_loop(
	ApplicationState& app_state,
	ActivityChat& state
)
{
	// uint32_t selected_peer_id;
	// std::optional<chatty::peer> selected_peer_info = std::nullopt;
	
	std::unique_ptr<std::string> modal_text_content = nullptr;
	
	size_t selected = 0;
	// std::vector<message_to_render> selected_peer_messages;
	
	// Example variables
	static char input[1024] = ""; // multi-line input buffer
	static ImGuiInputTextFlags input_flags = ImGuiInputTextFlags_EnterReturnsTrue;
	
	float left_width = 200.0f; // width for the peers list
	// Begin UI
	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
	ImGui::Begin("Chat", nullptr,
				 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
				 ImGuiWindowFlags_NoCollapse);
	
	ImGui::BeginChild("Peers", ImVec2(left_width, 0), true);
	
	// Show all peers
	if (ImGui::BeginListBox("##MyListBox", ImVec2(-FLT_MIN, 300.0f))) {
		for (int i = 0; i < app_state.peers_.size(); i++) {
			if (app_state.self_id_ == app_state.peers_[i].id) {
				continue;
			}
			
			bool is_selected = (selected == i);
			
			if (ImGui::Selectable(app_state.peers_[i].name.c_str(), is_selected)) {
				selected = i;
				uint32_t selected_peer_id = app_state.peers_[i].id;
				shark::log::debug("selected: {}:{}", i, app_state.peers_[i].id);
				
				state.selected_peer_messages.clear();
				for (auto &g : app_state.uni_db_->get_messages_for_peer(
					selected_peer_id, app_state.self_id_))
				{
					state.selected_peer_messages.push_back(
					message_to_render{.sender_id = g.sender_id,
						.reader_id = g.reader_id,
						.content = g.content});
					/*
					*	shark::log::debug("{} + {}->{}:{}", self_id, g.sender_id,
					*	g.reader_id, g.content);
					*/
				}
					
					state.selected_peer_info =
					app_state.uni_db_->get_peer_by_id(selected_peer_id);
			}
			
			// Set initial focus on selected item
			if (is_selected) {
				ImGui::SetItemDefaultFocus();
			}
		}
		ImGui::EndListBox();
	}
	
	if (ImGui::Button("Add Peer")) {
		ActivityPeerEditor peer_editor { ActivityPeerEditor::Mode::create };
	}
	
	static bool confirm_delete = false;
	
	if (ImGui::Button("Remove Peer")) {
		confirm_delete = true;
	}
	
	if (confirm_delete) {
		ImGui::OpenPopup("Confirm Delete");
	}
	
	if (state.selected_peer_info.has_value() &&
		ImGui::BeginPopupModal(
			"Confirm Delete", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
	{
		ImGui::Text("Delete this peer and all messages?");
		ImGui::Separator();
		
		if (ImGui::Button("Yes")) {
			app_state.uni_db_->remove_peer(state.selected_peer_info.value().id);
			app_state.peers_ = app_state.uni_db_->get_all_peers();
			state.selected_peer_messages.clear();
			state.selected_peer_info = std::nullopt;
			confirm_delete = false;
			ImGui::CloseCurrentPopup();
		}
		
		ImGui::SameLine();
		
		if (ImGui::Button("Cancel")) {
			confirm_delete = false;
			ImGui::CloseCurrentPopup();
		}
		
		ImGui::EndPopup();
	}
	
	ImGui::EndChild(); // ChatArea
	ImGui::SameLine();
	
	// Right panel: chat log + input
	ImGui::BeginChild("ChatArea", ImVec2(0, 0),
						false); // 0 width = remaining space
	// Input box
	
	float input_height = 80;
	float reserved = input_height + ImGui::GetFrameHeight() +
	ImGui::GetStyle().ItemSpacing.y * 2;
	// Scrollable log
	ImGui::BeginChild("log", ImVec2(0, -reserved), true);
	
	if (state.selected_peer_info.has_value()) {
		for (const auto &msg : state.selected_peer_messages) {
			
			bool is_user = msg.sender_id == app_state.self_id_;
			
			if (is_user) {
				ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Me:");
			} else {
				ImGui::TextColored(
					ImVec4(1.0f, 0.6f, 0.4f, 1.0f), "%s:",
					state.selected_peer_info.value().name.c_str()
				);
			}
			
			ImGui::SameLine();
			
			ImGui::PushTextWrapPos();
			
			if (msg.tmp_stream != nullptr) {
				std::scoped_lock _(msg.tmp_stream->tmp_mtx_stream);
				
				const std::string& msg1 = msg.tmp_stream->tmp_stream;
				ImGui::TextUnformatted(msg1.c_str());
				
				if (ImGui::IsItemHovered() &&
					ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
				{
					modal_text_content = std::make_unique<std::string>(std::string(msg.content));
					// ImGui::SetClipboardText(msg1.c_str());
				}
			} else {
				ImGui::TextUnformatted(msg.content.c_str());
				if (ImGui::IsItemHovered() &&
					ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
				{
					modal_text_content = std::make_unique<std::string>(std::string(msg.content));
					// ImGui::SetClipboardText(msg.content.c_str());
				}
			}
			
			ImGui::PopTextWrapPos();
			
			ImGui::Spacing();
		}
	}
	
	ImGui::EndChild();
	
	ImGui::PushItemWidth(-1); // full width
	bool send_pressed = ImGui::InputTextMultiline(
		"##input", input, sizeof(input),
		ImVec2(0, input_height), // height 80 pixels
		ImGuiInputTextFlags_EnterReturnsTrue
	);
	ImGui::PopItemWidth();
	// Send button
	// ImGui::SameLine();
	
	auto lam_load = [&] {
		std::string input_str = input;
		app_state.uni_db_->insert_message(
			app_state.self_id_, state.selected_peer_info.value().id,
			input_str
		);
		state.selected_peer_messages.emplace_back(
			message_to_render{
				.sender_id = app_state.self_id_,
				.reader_id = state.selected_peer_info.value().id,
				.content = input_str}
		);
		input[0] = 0; // clear input
	};
	auto lam_signal = [&] {
		// ImGuiInputTextFlags_EnterReturnsTrue
		// If current peer is set, then the shared string will be dropped so
		// only
		//   send_message_llama keeps it.
		std::shared_ptr<chatty::dynamic_to_render> tmp_stream =
		std::make_shared<chatty::dynamic_to_render>();
		
		tmp_stream->on_completed =
			[
				selected_peer_id = state.selected_peer_info.value().id,
				&app_state
			] (const std::string& reply) {
				app_state.uni_db_->insert_message(
					selected_peer_id, app_state.self_id_, reply
				);
			};
		
		shark::log::debug("Starting Thread to call send_message_llama");
		
		app_state.multi_executor_->push(
			[
				selected_peer_id = state.selected_peer_info.value().id,
				tmp_stream, &app_state
			] () -> void {
				auto payload = prepare_payload(
					*app_state.uni_db_, selected_peer_id, app_state.self_id_
				);
				chatty::send_message_llama(
					app_state, payload, tmp_stream
				);
			}
		);
		
		state.selected_peer_messages.emplace_back(
			message_to_render{
				.sender_id = state.selected_peer_info.value().id,
				.reader_id = app_state.self_id_,
				.tmp_stream = tmp_stream
			}
		);
		// selected_peer_messages.emplace_back(chatty::message{.sender_id =
		// self_id, .reader_id = selected_peer_id })
	};
	
	if (state.selected_peer_info.has_value()) {
		if (ImGui::Button("Load")) {
			lam_load();
		}
		ImGui::SameLine();
		if (ImGui::Button("Signal")) {
			lam_signal();
		}
		ImGui::SameLine();
		if (ImGui::Button("Interrupt")) {
			for (auto it = state.selected_peer_messages.begin();
				it != state.selected_peer_messages.end(); ++it) {
				if (it->tmp_stream != nullptr) {
					std::scoped_lock _(it->tmp_stream->tmp_mtx_stream);
					if (
						it->tmp_stream->status ==
						chatty::dynamic_to_render::status::STREAMING)
					{
						it->tmp_stream->status =
						chatty::dynamic_to_render::status::INTERRUPTED;
						state.selected_peer_messages.erase(it);
					break;
					}
				}
			}
		}
		ImGui::SameLine();
		if (ImGui::Button("Remove Last Message")) {
			if (!state.selected_peer_messages.empty()) {
				{
					state.selected_peer_messages.pop_back();
					
					app_state.uni_db_->remove_last_message(
						state.selected_peer_info.value().id,
						app_state.self_id_
					);
				}
			}
		}
		ImGui::SameLine();
		if (ImGui::Button("Lorebook")) {
			// show_lorebook_window = true;
		}
		
		ImGui::SameLine();
		if (ImGui::Button("Edit Peer")) {
			ActivityPeerEditor peer_editor{ 
				ActivityPeerEditor::Mode::edit,
				state.selected_peer_info->id 
			};
			
			strncpy(
				peer_editor.name, state.selected_peer_info->name.c_str(),
				sizeof(peer_editor.name)
			);
			
			strncpy(
				peer_editor.card, state.selected_peer_info->card.c_str(),
				sizeof(peer_editor.card)
			);
			
			app_state.states_.emplace_back(
				std::make_unique<ActivityPeerEditor>(std::move(peer_editor))
			);
		}
		ImGui::SameLine();
		if (ImGui::Button("Dump Logs")) {
			boost::json::array pre;
			for (const auto &msg : state.selected_peer_messages) {
				boost::json::object preo;
				preo.emplace(
					"speaker", msg.sender_id == app_state.self_id_ ? 
					"Sensei" : state.selected_peer_info->name);
				preo.emplace("text", msg.content);
				pre.push_back(std::move(preo));
			}
			modal_text_content = std::make_unique<std::string>(
				boost::json::serialize(pre));
		}
	}
	
	ImGui::EndChild(); // ChatArea
	
	// Call every frame
	// RenderPeerEditorWindow(peer_editor, peers, uni_db, self_id);
	// RenderLorebookWindow(uni_db, state.selected_peer_info);
	ShowMessageBox("Modal Text", modal_text_content);
	ImGui::End();
}


int main()
{
	
	if (!glfwInit())
		return -1;

	const char *glsl_version = "#version 130";
	GLFWwindow *window =
	    glfwCreateWindow(1280, 720, "ImGui Example", NULL, NULL);
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1); // vsync

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	// static std::string log;

	// PeerEditorState peer_editor;
	std::unique_ptr<ApplicationState> state = std::make_unique<ApplicationState>();
	state->uni_config_ = std::make_unique<chatty::config>("chatty.json");
	shark::log::debug("chatty.json loaded");
	state->uni_db_ = std::make_unique<chatty::db>("chatty.db");
	shark::log::debug("chatty.db loaded");
	state->self_id_ = 0;
	state->peers_ = state->uni_db_->get_all_peers();
	state->multi_executor_ = std::make_unique<shark::async::multi_executor<void()>>(3);
	
	ImGuiIO &io = ImGui::GetIO();

	ImFontConfig cfg;
	cfg.OversampleH = 2;
	cfg.OversampleV = 2;
	cfg.PixelSnapH = true;

	ImFont *font = io.Fonts->AddFontFromFileTTF(
	    "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
	    22.0f, &cfg, io.Fonts->GetGlyphRangesDefault());
	
	
	// ActivityChat chat_state;
	
	while (!glfwWindowShouldClose(window)) {
		
		state->states_.erase(
			std::remove_if(
				state->states_.begin(), state->states_.end(),
				[](std::unique_ptr<ActivityState>& act) {
					return !act->open;
				}
			),
			state->states_.end()
		);
		
		if (state->states_.empty()) {
			state->states_.push_back(
				std::make_unique<ActivityState>( ActivityState{ Activitie::LOGIN } )
			);
		}
		
		std::unique_ptr<ActivityState>& last_activity = 
			state->states_.at(state->states_.size() - 1ULL);
		
		glfwPollEvents();
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::PushFont(font);
		
		switch(last_activity->activity_) {
			case Activitie::LOGIN: {
				// auto* ptr = dynamic_cast<ActivityChat*>(last_activity.get());
				RenderLoginPanel(*state, *last_activity);
			break; }
			case Activitie::CHAT: {
				auto* ptr = dynamic_cast<ActivityChat*>(last_activity.get());
				chat_loop(*state, *ptr);
			break; }
			case Activitie::PEER_EDITOR: {
				auto* ptr = dynamic_cast<ActivityPeerEditor*>(last_activity.get());
				RenderPeerEditorWindow(*state, *ptr);
			break; }
			default: {
				shark::raise(
					"Invalid Last Activitie Sort: {}",
					static_cast<int>(last_activity->activity_)
				);
			break; }
		}
		
		ImGui::PopFont();
		// ImGui::End();

		// Render
		ImGui::Render();
		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		glClear(GL_COLOR_BUFFER_BIT);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glfwSwapBuffers(window);
	}

	shark::log::debug("Application is shutting down");
	state->shutdown();
	
	// Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwDestroyWindow(window);
	glfwTerminate();
	
	state = nullptr;
	
	shark::log::kill();
	// shark::log::debug("Application closing");

	return 0;
}


