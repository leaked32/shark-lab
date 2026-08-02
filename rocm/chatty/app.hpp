/*
 * Project: shark-lab
 * Repository: https://github.com/leaked32/shark-lab
 *
 * File: chatty/app.hpp
 *
 * License: MIT
 */

#pragma once
#include <memory>
#include "db.hpp"
#include "shark/shark.hpp"

enum class Activitie {
	INVALID, LOGIN, CHAT, PEER_EDITOR, LOREBOOK,
};

struct message_to_render {
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
	
	ActivityState(Activitie activitie) : activity_(activitie) 
	{ }
	
	ActivityState(Activitie activitie, uint32_t selected_peer_id) :
		activity_(activitie), selected_peer_id_(selected_peer_id)
	{ }
};

struct ActivityPeerEditor : ActivityState
{
	enum class Mode { create, edit } mode = Mode::create;
	char name[128] = "";
	char card[4096] = "";
	
	ActivityPeerEditor(Mode mode, uint32_t selected_peer_id = 0) :
		mode(mode), ActivityState{ Activitie::PEER_EDITOR, selected_peer_id }
	{
		
	}
};

struct ActivityChat : ActivityState {
	std::optional<chatty::peer> selected_peer_info;
	std::vector<message_to_render> selected_peer_messages;
	char input[1024] = {};
	
	ActivityChat() : ActivityState{ Activitie::CHAT }
	{
		
	}
};

struct ApplicationState {
	// Activities activitie;
	// bool open = false;
	// bool show_lorebook_window;
	
	std::vector<std::unique_ptr<ActivityState>> states_ = {};
	
	int editing_peer_id_ = -1;
	std::vector<chatty::peer> peers_ = {};
	std::unique_ptr<chatty::db> uni_db_ = nullptr;
	
	std::unique_ptr<chatty::config> uni_config_ = nullptr;
	
	uint32_t self_id_ = 0;
	// std::vector<std::thread> workers_;
	
	// 
	std::unique_ptr<shark::async::multi_executor<void()>> multi_executor_;
	
	void shutdown() {
		shutting_down_ = true;
		multi_executor_->shutdown();
	}
	bool shutting_down() const {
		return shutting_down_;
	}
private:
	std::atomic_bool shutting_down_ = false;
};


void RenderPeerEditorWindow(
	ApplicationState& app_state,
	ActivityPeerEditor& peer_state
);

void RenderLorebookWindow(
	ApplicationState& app_state,
	std::optional<chatty::peer> selected_peer_info
);

void ShowMessageBox(const char* title, std::unique_ptr<std::string>& message);
