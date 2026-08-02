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
#include <GLFW/glfw3.h>
#include <boost/json.hpp>

#include "shark/media.hpp"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
// #include <ranges>
#include "app.hpp"
#include "external.hpp"

void RenderLoginPanel(
	ApplicationState& app_state, ActivityState& act_state)
{
	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);

	ImGui::Begin("Login", nullptr,
				 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

	const auto peers = app_state.uni_db_->get_all_peers();
	auto selected_it = std::ranges::find(*peers, act_state.selected_peer_id_, &chatty::peer::id);
	const chatty::peer* selected_peer = selected_it == peers->end() ? nullptr : &*selected_it;

	if (act_state.selected_peer_id_ != 0U && selected_peer == nullptr) {
		act_state.selected_peer_id_ = 0U;
	}

	constexpr float left_width = 300.0f;
	ImGui::BeginChild("PeerList", ImVec2(left_width, 0), true);

	ImGui::Text("Select identity:");
	ImGui::Separator();

	if (ImGui::BeginListBox("##peers", ImVec2(-FLT_MIN, -300.F))) {
		for (const auto& peer : *peers) {
			const bool active = act_state.selected_peer_id_ == peer.id;
			const std::string label = peer.name + "##" + std::to_string(peer.id);

			if (ImGui::Selectable(label.c_str(), active)) {
				act_state.selected_peer_id_ = peer.id;
				selected_peer = &peer;
			}

			if (active) {
				ImGui::SetItemDefaultFocus();
			}
		}

		ImGui::EndListBox();
	}

	if (ImGui::Button("Add Peer")) {
		app_state.states_.emplace_back(
			std::make_unique<ActivityPeerEditor>(ActivityPeerEditor::Mode::create));
	}

	ImGui::EndChild();
	ImGui::SameLine();
	ImGui::BeginChild("PeerPreview", ImVec2(0, 0), true);

	if (selected_peer != nullptr) {
		ImGui::Text("Name: %s", selected_peer->name.c_str());
		ImGui::Separator();

		if (!selected_peer->card.empty()) {
			ImGui::TextWrapped("%s", selected_peer->card.c_str());
		}
		else {
			ImGui::TextDisabled("No card");
		}

		ImGui::Spacing();
		if (ImGui::Button("Login", ImVec2(120, 40))) {
			app_state.self_id_ = selected_peer->id;
			act_state.open = false;
			app_state.states_.emplace_back(std::make_unique<ActivityChat>());
		}
	}
	else {
		ImGui::TextDisabled("Select a peer");
	}

	ImGui::EndChild();
	ImGui::End();
}

void RenderChatPanel(
	ApplicationState& app_state, ActivityChat& state)
{
	const auto peers = app_state.uni_db_->get_all_peers();
	auto selected_it = std::ranges::find(*peers, state.selected_peer_id_, &chatty::peer::id);
	const chatty::peer* selected_peer = selected_it == peers->end() ? nullptr : &*selected_it;

	if (state.selected_peer_id_ != 0U && selected_peer == nullptr) {
		state.selected_peer_id_ = 0U;
		state.selected_peer_messages.clear();
	}

	constexpr float left_width = 200.0f;
	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
	ImGui::Begin("Chat", nullptr,
				 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

	ImGui::BeginChild("Peers", ImVec2(left_width, 0), true);

	if (ImGui::BeginListBox("##MyListBox", ImVec2(-FLT_MIN, 300.0f))) {
		for (const auto& peer : *peers) {
			if (app_state.self_id_ == peer.id) {
				continue;
			}

			const bool is_selected = state.selected_peer_id_ == peer.id;
			const std::string label = peer.name + "##" + std::to_string(peer.id);

			if (ImGui::Selectable(label.c_str(), is_selected)) {
				state.selected_peer_id_ = peer.id;
				selected_peer = &peer;
				shark::log::debug("selected peer: {}", peer.id);

				state.selected_peer_messages.clear();
				for (const auto& message :
					 app_state.uni_db_->get_messages_for_peer(peer.id, app_state.self_id_))
				{
					state.selected_peer_messages.push_back(
						message_to_render{.sender_id = message.sender_id,
										  .reader_id = message.reader_id,
										  .content = message.content});
				}
			}

			if (is_selected) {
				ImGui::SetItemDefaultFocus();
			}
		}
		ImGui::EndListBox();
	}

	if (ImGui::Button("Add Peer")) {
		app_state.states_.emplace_back(
			std::make_unique<ActivityPeerEditor>(ActivityPeerEditor::Mode::create));
	}

	static bool confirm_delete = false;
	if (ImGui::Button("Remove Peer") && selected_peer != nullptr) {
		confirm_delete = true;
	}

	if (confirm_delete && selected_peer == nullptr) {
		confirm_delete = false;
	}

	if (confirm_delete) {
		ImGui::OpenPopup("Confirm Delete");
	}

	if (selected_peer != nullptr &&
		ImGui::BeginPopupModal("Confirm Delete", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
	{
		ImGui::Text("Delete this peer and all messages?");
		ImGui::Separator();

		if (ImGui::Button("Yes")) {
			app_state.uni_db_->remove_peer(state.selected_peer_id_);
			state.selected_peer_id_ = 0U;
			state.selected_peer_messages.clear();
			selected_peer = nullptr;
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

	ImGui::EndChild();
	ImGui::SameLine();
	ImGui::BeginChild("ChatArea", ImVec2(0, 0), false);

	constexpr float input_height = 80.0f;
	const float reserved =
		input_height + ImGui::GetFrameHeight() + ImGui::GetStyle().ItemSpacing.y * 2;
	ImGui::BeginChild("log", ImVec2(0, -reserved), true);

	if (selected_peer != nullptr) {
		for (const auto& msg : state.selected_peer_messages) {
			const bool is_user = msg.sender_id == app_state.self_id_;

			if (is_user) {
				ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Me:");
			}
			else {
				ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.4f, 1.0f),
								   "%s:", selected_peer->name.c_str());
			}

			ImGui::SameLine();
			ImGui::PushTextWrapPos();

			if (msg.tmp_stream != nullptr) {
				std::scoped_lock lock(msg.tmp_stream->tmp_mtx_stream);
				const std::string& streamed_text = msg.tmp_stream->tmp_stream;
				ImGui::TextUnformatted(streamed_text.c_str());

				if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
					app_state.states_.emplace_back(
						std::make_unique<ActivityModalText>(streamed_text, "Inspect"));
					ImGui::SetClipboardText(streamed_text.c_str());
				}
			}
			else {
				ImGui::TextUnformatted(msg.content.c_str());
				if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
					app_state.states_.emplace_back(
						std::make_unique<ActivityModalText>(msg.content, "Inspect"));
					ImGui::SetClipboardText(msg.content.c_str());
				}
			}

			ImGui::PopTextWrapPos();
			ImGui::Spacing();
		}
	}

	ImGui::EndChild();

	ImGui::PushItemWidth(-1);
	const bool send_pressed =
		ImGui::InputTextMultiline("##input", state.input, sizeof(state.input),
								  ImVec2(0, input_height), ImGuiInputTextFlags_EnterReturnsTrue);
	ImGui::PopItemWidth();

	auto remove_failed_streams = [&messages = state.selected_peer_messages]()
	{
		std::erase_if(messages, [](const message_to_render& message)
					  { return message.tmp_stream != nullptr && message.tmp_stream->failed; });
	};
	remove_failed_streams();

	if (selected_peer != nullptr) {
		const uint32_t selected_peer_id = state.selected_peer_id_;

		auto send_input = [&]
		{
			if (state.input[0] == '\0') {
				return;
			}

			std::string input_text = state.input;
			app_state.uni_db_->insert_message(app_state.self_id_, selected_peer_id, input_text);
			state.selected_peer_messages.emplace_back(
				message_to_render{.sender_id = app_state.self_id_,
								  .reader_id = selected_peer_id,
								  .content = std::move(input_text)});
			state.input[0] = '\0';
		};

		auto signal = [&]
		{
			auto tmp_stream = std::make_shared<chatty::dynamic_to_render>();
			tmp_stream->on_completed = [selected_peer_id, &app_state](const std::string& reply)
			{ app_state.uni_db_->insert_message(selected_peer_id, app_state.self_id_, reply); };

			shark::log::debug("Starting Thread to call send_message_llama");
			app_state.multi_executor_->push(
				[selected_peer_id, tmp_stream, &app_state]()
				{
					auto payload =
						chatty::prepare_payload(app_state, selected_peer_id, app_state.self_id_);
					if (!payload.has_value()) {
						tmp_stream->failed = true;
						return;
					}

					chatty::send_message_llama(app_state, *payload, tmp_stream);
				});

			state.selected_peer_messages.emplace_back(
				message_to_render{.sender_id = selected_peer_id,
								  .reader_id = app_state.self_id_,
								  .tmp_stream = std::move(tmp_stream)});
		};

		const bool load_pressed = ImGui::Button("Load");
		if (send_pressed || load_pressed) {
			send_input();
		}
		ImGui::SameLine();
		if (ImGui::Button("Signal")) {
			signal();
		}
		ImGui::SameLine();
		if (ImGui::Button("Interrupt")) {
			for (auto it = state.selected_peer_messages.begin();
				 it != state.selected_peer_messages.end(); ++it)
			{
				if (it->tmp_stream == nullptr) {
					continue;
				}

				std::scoped_lock lock(it->tmp_stream->tmp_mtx_stream);
				if (it->tmp_stream->status == chatty::dynamic_to_render::status::STREAMING) {
					it->tmp_stream->status = chatty::dynamic_to_render::status::INTERRUPTED;
					state.selected_peer_messages.erase(it);
					break;
				}
			}
		}
		ImGui::SameLine();
		if (ImGui::Button("Remove Last Message") && !state.selected_peer_messages.empty()) {
			state.selected_peer_messages.pop_back();
			app_state.uni_db_->remove_last_message(selected_peer_id, app_state.self_id_);
		}
		ImGui::SameLine();
		if (ImGui::Button("Lorebook")) {
			// show_lorebook_window = true;
		}
		ImGui::SameLine();
		if (ImGui::Button("Edit Peer")) {
			ActivityPeerEditor peer_editor{ActivityPeerEditor::Mode::edit, selected_peer_id};
			std::strncpy(peer_editor.name, selected_peer->name.c_str(),
						 sizeof(peer_editor.name) - 1U);
			std::strncpy(peer_editor.card, selected_peer->card.c_str(),
						 sizeof(peer_editor.card) - 1U);
			peer_editor.name[sizeof(peer_editor.name) - 1U] = '\0';
			peer_editor.card[sizeof(peer_editor.card) - 1U] = '\0';
			app_state.states_.emplace_back(
				std::make_unique<ActivityPeerEditor>(std::move(peer_editor)));
		}
		ImGui::SameLine();
		if (ImGui::Button("Dump Logs")) {
			boost::json::array logs;
			for (const auto& msg : state.selected_peer_messages) {
				boost::json::object item;
				item.emplace("speaker",
							 msg.sender_id == app_state.self_id_ ? "Sensei" : selected_peer->name);
				item.emplace("text", msg.content);
				logs.push_back(std::move(item));
			}
			app_state.states_.emplace_back(
				std::make_unique<ActivityModalText>(boost::json::serialize(logs), "Dump Logs"));
		}
	}

	ImGui::EndChild();
	ImGui::End();
}

void setup_chatty_theme(
	const chatty::config& cfg)
{
	if (cfg.dark_mode_) {
		ImGui::StyleColorsDark();
	}

	auto& style = ImGui::GetStyle();

	style.WindowRounding = 10.0f;
	style.FrameRounding = 8.0f;
	style.GrabRounding = 8.0f;

	style.WindowPadding = ImVec2(12, 12);
	style.FramePadding = ImVec2(8, 6);

	style.Colors[ImGuiCol_WindowBg] = cfg.window_bg_.get<ImVec4>();
	style.Colors[ImGuiCol_FrameBg] = cfg.frame_bg_.get<ImVec4>();
	style.Colors[ImGuiCol_Button] = cfg.button_.get<ImVec4>();
	style.Colors[ImGuiCol_ButtonHovered] = cfg.button_hovered_.get<ImVec4>();
	style.Colors[ImGuiCol_ButtonActive] = cfg.button_active_.get<ImVec4>();
	style.Colors[ImGuiCol_Text] = cfg.text_.get<ImVec4>();
}
void draw_background(
	const shark::media::texture& tex)
{
	auto screen = ImGui::GetIO().DisplaySize;

	float image_ratio = static_cast<float>(tex.width) / static_cast<float>(tex.height);

	float screen_ratio = screen.x / screen.y;

	ImVec2 uv0(0, 0);
	ImVec2 uv1(1, 1);

	if (image_ratio > screen_ratio) {
		float visible = screen_ratio / image_ratio;
		float offset = (1.0f - visible) * 0.5f;

		uv0.x = offset;
		uv1.x = 1.0f - offset;
	}
	else {
		float visible = image_ratio / screen_ratio;
		float offset = (1.0f - visible) * 0.5f;

		uv0.y = offset;
		uv1.y = 1.0f - offset;
	}

	ImGui::SetNextWindowPos(ImVec2(0, 0));

	ImGui::SetNextWindowSize(screen);

	ImGui::Begin("##background", nullptr,
				 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
					 ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoBackground);

	ImGui::Image((ImTextureID)(intptr_t)tex.id, screen, uv0, uv1);

	ImGui::End();
}

int main(
	int argc, char** argv)
{
	if (!glfwInit()) {
		return -1;
	}

	const char* glsl_version = "#version 130";
	GLFWwindow* window = glfwCreateWindow(1280, 720, "ImGui Example", NULL, NULL);
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1); // vsync

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	// 	namespace fs = std::filesystem;
	//
	// 	bool exists = fs::exists("chatty_background.png");
	// 	GLuint background_texture = 0;
	//
	// 	if (fs::exists("chatty_background.png"))
	// 	{
	// 		background_texture =
	// 		ImGui::loadTe("chatty_background.png");
	// 	}
	// static std::string log;

	// PeerEditorState peer_editor;
	std::unique_ptr<ApplicationState> state = std::make_unique<ApplicationState>();
	state->uni_config_ = std::make_unique<chatty::config>("chatty.json");
	shark::log::debug("chatty.json loaded");
	state->uni_db_ = std::make_unique<chatty::db>("chatty.db");
	shark::log::debug("chatty.db loaded");
	state->self_id_ = 0;
	state->multi_executor_ = std::make_unique<shark::async::multi_executor<void()>>(3);

	setup_chatty_theme(*state->uni_config_);

	ImGuiIO& io = ImGui::GetIO();

	ImFontConfig cfg;
	cfg.OversampleH = 2;
	cfg.OversampleV = 2;
	cfg.PixelSnapH = true;

	ImFont* font = nullptr;
	if (!state->uni_config_->font_.empty()) {
		std::string abs_font_path = shark::media::find_system_font(state->uni_config_->font_);
		font = io.Fonts->AddFontFromFileTTF(abs_font_path.c_str(), 22.0f, &cfg,
											io.Fonts->GetGlyphRangesDefault());
	}

	std::optional<shark::media::texture> background_texture;

	if (std::filesystem::exists(state->uni_config_->background_path_)) {
		background_texture = shark::media::load_texture(state->uni_config_->background_path_);
	}
	// ActivityChat chat_state;

	while (!glfwWindowShouldClose(window)) {
		std::erase_if(state->states_,
					  [](std::unique_ptr<ActivityState>& act) { return !act->open; });

		if (state->states_.empty()) {
			state->states_.push_back(
				std::make_unique<ActivityState>(ActivityState{Activitie::LOGIN}));
		}

		std::unique_ptr<ActivityState>& last_activity =
			state->states_.at(state->states_.size() - 1ULL);

		glfwPollEvents();
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// Background Image is invoked here immediately.
		if (background_texture.has_value()) {
			draw_background(background_texture.value());
		}

		if (font) {
			ImGui::PushFont(font);
		}

		switch (last_activity->activity_) {
		case Activitie::LOGIN: {
			// auto* ptr = dynamic_cast<ActivityChat*>(last_activity.get());
			RenderLoginPanel(*state, *last_activity);
			break;
		}
		case Activitie::CHAT: {
			auto* ptr = dynamic_cast<ActivityChat*>(last_activity.get());
			RenderChatPanel(*state, *ptr);
			break;
		}
		case Activitie::PEER_EDITOR: {
			auto* ptr = dynamic_cast<ActivityPeerEditor*>(last_activity.get());
			RenderPeerEditorWindow(*state, *ptr);
			break;
		}
		case Activitie::MODAL_TEXT: {
			auto* ptr = dynamic_cast<ActivityModalText*>(last_activity.get());
			RenderModalText(*state, *ptr);
			break;
		}
		default: {
			shark::raise("Invalid Last Activitie Sort: {}",
						 static_cast<int>(last_activity->activity_));
			break;
		}
		}

		if (font) {
			ImGui::PopFont();
		}
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
