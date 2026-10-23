"""FootyGPT Streamlit frontend.

Run with: streamlit run frontend/app.py

This file only builds the user interface. All AI work and saved conversations
live in the FastAPI backend, which this app calls over HTTP. Streamlit session
state holds only what is needed to draw the page; the database is the source
of truth for saved conversations.
"""

import streamlit as st

try:
    from frontend import api_client
except ImportError:  # running as `streamlit run frontend/app.py`
    import api_client  # type: ignore[no-redef]

TITLE = "FootyGPT — Your AI Football Companion"
DESCRIPTION = (
    "Ask FootyGPT anything about football — rules, tactics, formations, "
    "players, and football history. Answers are explained in plain language, "
    "so they're easy to follow even if you're new to the game."
)


def _init_state() -> None:
    """Create the session state keys on first run."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("confirm_delete_id", None)


def _start_new_conversation() -> None:
    """Clear the page and begin a fresh conversation.

    The conversation is created by the backend on the first question, so no
    empty conversation appears in the sidebar.
    """
    st.session_state.messages = []
    st.session_state.conversation_id = None
    st.session_state.confirm_delete_id = None


def _open_conversation(conversation_id: int) -> None:
    """Load a saved conversation's messages from the backend."""
    try:
        conversation = api_client.get_conversation(conversation_id)
    except api_client.APIError as error:
        st.session_state["sidebar_error"] = error.message
        return

    st.session_state.conversation_id = conversation["id"]
    st.session_state.messages = [
        {"role": message["role"], "content": message["content"]}
        for message in conversation["messages"]
    ]
    st.session_state.confirm_delete_id = None


def _confirm_delete(conversation_id: int) -> None:
    """Delete a conversation after the user has confirmed."""
    try:
        api_client.delete_conversation(conversation_id)
    except api_client.APIError as error:
        st.session_state["sidebar_error"] = error.message
        return

    if st.session_state.conversation_id == conversation_id:
        _start_new_conversation()
    # Force a rerun so the sidebar list is fetched again without the deleted row.
    st.rerun()


def _render_message(role: str, content: str) -> None:
    """Show one chat bubble."""
    with st.chat_message(role):
        st.markdown(content)


def _render_sidebar() -> None:
    """Draw saved conversations and their controls."""
    with st.sidebar:
        st.header("Conversations")

        if st.button("＋ New conversation", use_container_width=True):
            _start_new_conversation()

        error = st.session_state.pop("sidebar_error", None)
        if error:
            st.error(error)

        try:
            conversations = api_client.list_conversations()
        except api_client.APIError as error:
            st.error(error.message)
            conversations = []

        if not conversations:
            st.caption("No saved conversations yet.")

        for conversation in conversations:
            conversation_id = conversation["id"]
            label = conversation["title"] or "Untitled conversation"
            if conversation_id == st.session_state.conversation_id:
                label = f"▶ {label}"

            if st.button(label, key=f"open-{conversation_id}", use_container_width=True):
                _open_conversation(conversation_id)
                st.rerun()

        current_id = st.session_state.conversation_id
        if current_id is not None:
            st.divider()
            if st.session_state.confirm_delete_id == current_id:
                st.warning("Delete this conversation permanently?")
                confirm_col, cancel_col = st.columns(2)
                if confirm_col.button("Yes, delete", key="delete-yes", use_container_width=True):
                    _confirm_delete(current_id)
                if cancel_col.button("Cancel", key="delete-cancel", use_container_width=True):
                    st.session_state.confirm_delete_id = None
                    st.rerun()
            elif st.button("🗑 Delete conversation", key="delete-start", use_container_width=True):
                st.session_state.confirm_delete_id = current_id
                st.rerun()

        st.divider()
        st.caption(
            "Backend: 🟢 Online"
            if api_client.check_health()
            else "Backend: 🔴 Offline — start FastAPI and reload."
        )


def main() -> None:
    """Draw the page and handle new questions."""
    st.set_page_config(page_title="FootyGPT", page_icon="⚽", layout="centered")
    _init_state()

    st.title(TITLE)
    st.write(DESCRIPTION)

    _render_sidebar()

    for message in st.session_state.messages:
        _render_message(message["role"], message["content"])

    prompt = st.chat_input("Ask a football question…")
    if not prompt or not prompt.strip():
        return
    prompt = prompt.strip()

    st.session_state.messages.append({"role": "user", "content": prompt})
    _render_message("user", prompt)

    with st.spinner("Thinking…"):
        try:
            result = api_client.send_chat_message(
                prompt, conversation_id=st.session_state.conversation_id
            )
        except api_client.APIError as error:
            # The backend saves nothing when a request fails, so drop the
            # question from the local transcript too.
            st.session_state.messages.pop()
            st.error(error.message)
            return

    st.session_state.conversation_id = result["conversation_id"]
    answer = result["assistant_message"]["content"]
    st.session_state.messages.append({"role": "assistant", "content": answer})
    # Rerun so the sidebar picks up a newly created conversation.
    st.rerun()


if __name__ == "__main__":
    main()
