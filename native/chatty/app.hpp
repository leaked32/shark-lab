/*
 * Project: shark-lab
 * Repository: https://github.com/leaked32/shark-lab
 *
 * File: chatty/app.hpp
 *
 * License: MIT
 */

#pragma once
#include "db.hpp"
#include "shark/shark.hpp"
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

enum class Activitie { INVALID, LOGIN, CHAT, PEER_EDITOR, LOREBOOK, MODAL_TEXT };

struct message_to_render
{
	uint32_t sender_id;
	uint32_t reader_id;
	std::string content;
	std::shared_ptr<chatty::dynamic_to_render> tmp_stream = nullptr;
};

struct ActivityState
{
	bool open = true;
	uint32_t selected_peer_id_ = 0;
	Activitie activity_ = Activitie::INVALID;

	virtual ~ActivityState() = default;

	ActivityState(
		Activitie activitie) : activity_(activitie)
	{}

	ActivityState(
		Activitie activitie, uint32_t selected_peer_id) :
		activity_(activitie), selected_peer_id_(selected_peer_id)
	{}
};

struct ActivityPeerEditor : ActivityState
{
	enum class Mode { create, edit } mode = Mode::create;
	char name[128] = "";
	char card[4096] = "";

	ActivityPeerEditor(
		Mode mode, uint32_t selected_peer_id = 0) :
		mode(mode), ActivityState{Activitie::PEER_EDITOR, selected_peer_id}
	{}
};

struct ActivityChat : ActivityState
{
	std::vector<message_to_render> selected_peer_messages;
	char input[1024] = {};

	ActivityChat() : ActivityState{Activitie::CHAT} {}
};

struct ActivityModalText : ActivityState
{
	ActivityModalText(
		const std::string& text, const std::string& title) :
		ActivityState{Activitie::MODAL_TEXT}, text_{text}, title_(title)
	{}

	std::string text_;
	std::string title_;
};

struct ApplicationState
{
	// Activities activitie;
	// bool open = false;
	// bool show_lorebook_window;

	std::vector<std::unique_ptr<ActivityState>> states_ = {};
	std::unique_ptr<chatty::db> uni_db_ = nullptr;
	std::unique_ptr<chatty::config> uni_config_ = nullptr;

	uint32_t self_id_ = 0;
	// std::vector<std::thread> workers_;

	//
	std::unique_ptr<shark::async::multi_executor<void()>> multi_executor_;

	void shutdown()
	{
		shutting_down_ = true;
		multi_executor_->shutdown();
	}
	bool shutting_down() const
	{
		return shutting_down_;
	}
	void modal_text(const std::string& text, const std::string& title = "Modal Text")
	{
		ActivityModalText modal{text, title};
		this->states_.emplace_back(std::make_unique<ActivityModalText>(std::move(modal)));
	}

  private:
	std::atomic_bool shutting_down_ = false;
};

void RenderPeerEditorWindow(ApplicationState& app_state, ActivityPeerEditor& peer_state);

void RenderLorebookWindow(ApplicationState& app_state, uint32_t selected_peer_id);

void RenderModalText(ApplicationState& app_state, ActivityModalText& modal_text);
