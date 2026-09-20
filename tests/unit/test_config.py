"""Tests for TOML configuration loading."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchtable.config import AgentConfig, RunConfig, load_config
from benchtable.errors import ConfigurationError


class TestLoadConfig:
    @pytest.mark.parametrize(
        ("provider", "fields", "expected"),
        [
            ("openai_compatible", {}, {"api_key_env": "OPENAI_API_KEY"}),
            (
                "gigachat",
                {"credential_env": "GIGA_ENV", "scope": "GIGACHAT_API_PERS"},
                {"credential_env": "GIGA_ENV", "scope": "GIGACHAT_API_PERS"},
            ),
            (
                "yandex_ai_studio",
                {"credential_env": "YC_ENV", "folder_id": "folder-123"},
                {
                    "credential_env": "YC_ENV",
                    "folder_id": "folder-123",
                    "credential_kind": "oauth",
                },
            ),
        ],
    )
    def test_provider_configuration_defaults(
        self,
        tmp_path: Path,
        provider: str,
        fields: dict[str, str],
        expected: dict[str, str],
    ) -> None:
        lines = [f'provider = "{provider}"', 'model = "provider-model"']
        lines.extend(f'{key} = "{value}"' for key, value in fields.items())
        cfg_path = tmp_path / "provider.toml"
        cfg_path.write_text(
            '[run]\ngame = "tiny"\nmatches = 1\n\n[[agents]]\nid = "a"\n'
            + "\n".join(lines)
            + "\n"
        )

        agent = load_config(cfg_path).agents[0]

        assert agent.provider == provider
        for field, value in expected.items():
            assert getattr(agent, field) == value

    @pytest.mark.parametrize(
        "agent_fields",
        [
            'provider = "gigachat"\nmodel = "provider-model"',
            'provider = "yandex_ai_studio"\nmodel = "provider-model"\n'
            'credential_env = "YC_ENV"',
            'provider = "yandex_ai_studio"\nmodel = "provider-model"\n'
            'folder_id = "folder"\ncredential_kind = "iam"\n'
            'credential_env = "YC_ENV"',
            'provider = "gigachat"\nmodel = "provider-model"\n'
            'credential_env = "GIGA_ENV"\nscope = "S"\n'
            'api_key_env = "OPENAI_API_KEY"',
            'provider = "openai_compatible"\nmodel = "provider-model"\n'
            'credential_env = "GIGA_ENV"',
        ],
    )
    def test_rejects_invalid_provider_field_combinations(
        self, tmp_path: Path, agent_fields: str
    ) -> None:
        cfg_path = tmp_path / "invalid-provider.toml"
        cfg_path.write_text(
            '[run]\ngame = "tiny"\nmatches = 1\n\n[[agents]]\nid = "a"\n'
            + agent_fields
            + "\n"
        )

        with pytest.raises(ConfigurationError):
            load_config(cfg_path)

    def test_valid_toml_parses(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 5
            seed = 42
            max_turns = 100
            max_invalid_attempts = 2

            [[agents]]
            id = "player-1"
            role = "player"
            model = "gpt-4o"
            api_key_env = "OPENAI_API_KEY"
        """)
        )
        cfg = load_config(cfg_path)
        assert cfg.run.game == "tiny"
        assert cfg.run.matches == 5
        assert cfg.run.seed == 42
        assert cfg.run.max_turns == 100
        assert len(cfg.agents) == 1
        assert cfg.agents[0].id == "player-1"
        assert cfg.agents[0].model == "gpt-4o"
        assert cfg.agents[0].api_key_env == "OPENAI_API_KEY"

    def test_default_api_key_env(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [[agents]]
            id = "player-1"
            role = "player"
            model = "gpt-4o"
        """)
        )
        cfg = load_config(cfg_path)
        assert cfg.agents[0].api_key_env == "OPENAI_API_KEY"

    def test_gigachat_scope_defaults_to_personal_api_scope(
        self, tmp_path: Path
    ) -> None:
        cfg_path = tmp_path / "gigachat.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [[agents]]
            id = "player-1"
            provider = "gigachat"
            model = "GigaChat-3-Ultra"
            credential_env = "GIGA_ENV"
        """)
        )

        cfg = load_config(cfg_path)

        assert cfg.agents[0].scope == "GIGACHAT_API_PERS"

    def test_gigachat_tls_verification_can_be_disabled(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "gigachat.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [[agents]]
            id = "player-1"
            provider = "gigachat"
            model = "GigaChat-3-Ultra"
            credential_env = "GIGA_ENV"
            verify_ssl_certs = false
        """)
        )

        cfg = load_config(cfg_path)

        assert cfg.agents[0].verify_ssl_certs is False

    def test_optional_base_url(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [[agents]]
            id = "player-1"
            role = "player"
            model = "gpt-4o"
            base_url = "https://custom.api.com/v1"
            api_key_env = "MY_KEY"
        """)
        )
        cfg = load_config(cfg_path)
        assert cfg.agents[0].base_url == "https://custom.api.com/v1"

    def test_optional_max_completion_tokens(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [[agents]]
            id = "player-1"
            model = "gpt-4o"
            max_completion_tokens = 128
        """)
        )

        cfg = load_config(cfg_path)

        assert cfg.agents[0].max_completion_tokens == 128

    def test_max_completion_tokens_must_be_positive(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [[agents]]
            id = "player-1"
            model = "gpt-4o"
            max_completion_tokens = 0
        """)
        )

        with pytest.raises(ConfigurationError):
            load_config(cfg_path)

    @pytest.mark.parametrize("value", ["true", '"128"', "128.0"])
    def test_max_completion_tokens_rejects_coercion(
        self, tmp_path: Path, value: str
    ) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent(
                f"""\
                [run]
                game = "tiny"
                matches = 1

                [[agents]]
                id = "player-1"
                model = "gpt-4o"
                max_completion_tokens = {value}
            """
            )
        )

        with pytest.raises(ConfigurationError):
            load_config(cfg_path)

    def test_rejects_deprecated_max_tokens(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [[agents]]
            id = "player-1"
            model = "gpt-4o"
            max_tokens = 128
        """)
        )

        with pytest.raises(ConfigurationError):
            load_config(cfg_path)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("id", '""'),
            ("model", '""'),
            ("base_url", '""'),
            ("timeout", "0"),
        ],
    )
    def test_provider_boundary_fields_are_validated(
        self, tmp_path: Path, field: str, value: str
    ) -> None:
        cfg_path = tmp_path / "test.toml"
        id_value = value if field == "id" else '"player-1"'
        model_value = value if field == "model" else '"gpt-4o"'
        base_url_value = value if field == "base_url" else '"https://example.test/v1"'
        timeout_value = value if field == "timeout" else "30.0"
        cfg_path.write_text(
            textwrap.dedent(
                f"""\
                [run]
                game = "tiny"
                matches = 1

                [[agents]]
                id = {id_value}
                model = {model_value}
                base_url = {base_url_value}
                timeout = {timeout_value}
            """
            )
        )

        with pytest.raises(ConfigurationError):
            load_config(cfg_path)

    @pytest.mark.parametrize(
        "value",
        ["true", '"30"', '"   "', "nan", "inf", "-inf", "0", "-1"],
    )
    def test_timeout_rejects_invalid_values(self, tmp_path: Path, value: str) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent(
                f"""\
                [run]
                game = "tiny"
                matches = 1

                [[agents]]
                id = "player-1"
                model = "gpt-4o"
                timeout = {value}
            """
            )
        )

        with pytest.raises(ConfigurationError):
            load_config(cfg_path)

    @pytest.mark.parametrize("value", ["30", "30.5"])
    def test_timeout_accepts_positive_numeric_values(
        self, tmp_path: Path, value: str
    ) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent(
                f"""\
                [run]
                game = "tiny"
                matches = 1

                [[agents]]
                id = "player-1"
                model = "gpt-4o"
                timeout = {value}
            """
            )
        )

        cfg = load_config(cfg_path)

        assert cfg.agents[0].timeout == float(value)

    def test_agent_config_timeout_overflow_is_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError):
            AgentConfig(
                id="player-1",
                model="gpt-4o",
                timeout=10**1000,
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            (field, value)
            for field in [
                "matches",
                "seed",
                "max_turns",
                "max_invalid_attempts",
                "max_provider_retries",
            ]
            for value in ["true", '"1"', "1.0"]
        ],
    )
    def test_run_integer_fields_reject_coercion(
        self, tmp_path: Path, field: str, value: str
    ) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent(
                f"""\
                [run]
                game = "tiny"
                matches = 1
                {field} = {value}

                [[agents]]
                id = "player-1"
                model = "gpt-4o"
            """
            )
        )

        with pytest.raises(ConfigurationError):
            load_config(cfg_path)

    @pytest.mark.parametrize("field", ["id", "model", "base_url", "api_key_env"])
    def test_provider_strings_reject_whitespace_only(
        self, tmp_path: Path, field: str
    ) -> None:
        cfg_path = tmp_path / "test.toml"
        values = {
            "id": '"player-1"',
            "model": '"gpt-4o"',
            "base_url": '"https://example.test/v1"',
            "api_key_env": '"TEST_API_KEY"',
        }
        values[field] = '"   "'
        agent_fields = "\n".join(f"{name} = {value}" for name, value in values.items())
        cfg_path.write_text(
            textwrap.dedent(
                f"""\
                [run]
                game = "tiny"
                matches = 1

                [[agents]]
                {agent_fields}
            """
            )
        )

        with pytest.raises(ConfigurationError):
            load_config(cfg_path)

    def test_required_game_field(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            matches = 1

            [[agents]]
            id = "p1"
            role = "player"
            model = "gpt-4o"
        """)
        )
        with pytest.raises(ConfigurationError):
            load_config(cfg_path)

    def test_malformed_agent_entry_is_configuration_error(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            agents = [1]

            [run]
            game = "tiny"
            matches = 1
        """)
        )

        with pytest.raises(ConfigurationError):
            load_config(cfg_path)

    def test_directory_path_is_configuration_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError):
            load_config(tmp_path)

    def test_required_model_field(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [[agents]]
            id = "p1"
            role = "player"
        """)
        )
        with pytest.raises(ConfigurationError):
            load_config(cfg_path)

    def test_positive_matches_required(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 0

            [[agents]]
            id = "p1"
            role = "player"
            model = "gpt-4o"
        """)
        )
        with pytest.raises(ConfigurationError):
            load_config(cfg_path)

    def test_non_negative_budgets(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1
            max_invalid_attempts = -1

            [[agents]]
            id = "p1"
            role = "player"
            model = "gpt-4o"
        """)
        )
        with pytest.raises(ConfigurationError):
            load_config(cfg_path)

    def test_rejects_literal_api_key(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [[agents]]
            id = "p1"
            role = "player"
            model = "gpt-4o"
            api_key = "sk-secret"
        """)
        )
        with pytest.raises(ConfigurationError, match="api_key"):
            load_config(cfg_path)

    def test_rejects_nested_credential_without_exposing_value(
        self, tmp_path: Path
    ) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [run.game_config.credentials]
            Authorization = "sk-nested-secret"

            [[agents]]
            id = "p1"
            model = "gpt-4o"
        """)
        )

        with pytest.raises(ConfigurationError) as exc_info:
            load_config(cfg_path)

        assert "sk-nested-secret" not in str(exc_info.value)

    @pytest.mark.parametrize(
        "credential_key",
        [
            "api_token",
            "api_secret",
            "API.SECRET",
            "session-token",
            "SESSION_TOKEN",
            "client.token",
            "CLIENT_TOKEN",
            "proxy_authorization",
            "x_auth_token",
            "passwd",
            "secret-key",
            "bearer_token",
            "private_key",
            "refresh-token",
            "credential",
            "credentials",
        ],
    )
    def test_rejects_nested_credential_key_variants(
        self, tmp_path: Path, credential_key: str
    ) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent(
                f"""\
                [run]
                game = "tiny"
                matches = 1

                [run.game_config.nested]
                "{credential_key}" = "sk-variant-secret"

                [[agents]]
                id = "p1"
                model = "gpt-4o"
            """
            )
        )

        with pytest.raises(ConfigurationError) as exc_info:
            load_config(cfg_path)

        assert "sk-variant-secret" not in str(exc_info.value)

    def test_run_config_rejects_nested_credential_keys(self) -> None:
        with pytest.raises(ConfigurationError):
            RunConfig(
                game="tiny",
                matches=1,
                game_config={"nested": {"api_token": "sk-direct-secret"}},
            )

    def test_run_config_rejects_tuple_game_config_values(self) -> None:
        with pytest.raises(ValidationError):
            RunConfig(
                game="tiny",
                matches=1,
                game_config={"items": ("one", "two")},
            )

    def test_extra_field_error_does_not_echo_secret(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1
            unknown_field = "sk-extra-secret"

            [[agents]]
            id = "p1"
            model = "gpt-4o"
        """)
        )

        with pytest.raises(ConfigurationError) as exc_info:
            load_config(cfg_path)

        assert "sk-extra-secret" not in str(exc_info.value)

    def test_rejects_unknown_fields(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1
            unknown_field = "oops"

            [[agents]]
            id = "p1"
            role = "player"
            model = "gpt-4o"
        """)
        )
        with pytest.raises(ConfigurationError):
            load_config(cfg_path)

    def test_multiple_agents(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [[agents]]
            id = "p1"
            role = "player"
            model = "gpt-4o"

            [[agents]]
            id = "p2"
            role = "player"
            model = "gpt-4o-mini"
        """)
        )
        cfg = load_config(cfg_path)
        assert len(cfg.agents) == 2
        assert cfg.agents[0].id == "p1"
        assert cfg.agents[1].id == "p2"

    def test_duplicate_agent_ids_are_rejected_without_echoing_the_id(
        self, tmp_path: Path
    ) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [[agents]]
            id = "sk-duplicate-agent"
            model = "gpt-4o"

            [[agents]]
            id = "sk-duplicate-agent"
            model = "gpt-4o-mini"
        """)
        )

        with pytest.raises(ConfigurationError) as exc_info:
            load_config(cfg_path)

        assert "sk-duplicate-agent" not in str(exc_info.value)

    def test_game_config_passthrough(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [run.game_config]
            custom_param = "value"

            [[agents]]
            id = "p1"
            role = "player"
            model = "gpt-4o"
        """)
        )
        cfg = load_config(cfg_path)
        assert cfg.run.game_config == {"custom_param": "value"}

    def test_game_config_rejects_non_json_values(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [run.game_config]
            start_date = 2026-07-18

            [[agents]]
            id = "p1"
            role = "player"
            model = "gpt-4o"
        """)
        )

        with pytest.raises(ConfigurationError):
            load_config(cfg_path)

    @pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
    def test_game_config_rejects_non_finite_numbers(
        self, tmp_path: Path, value: str
    ) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent(
                f"""\
                [run]
                game = "tiny"
                matches = 1

                [run.game_config]
                score = {value}

                [[agents]]
                id = "p1"
                role = "player"
                model = "gpt-4o"
            """
            )
        )

        with pytest.raises(ConfigurationError):
            load_config(cfg_path)

    def test_invalid_utf8_configuration_is_configuration_error(
        self, tmp_path: Path
    ) -> None:
        cfg_path = tmp_path / "invalid.toml"
        cfg_path.write_bytes(b"[run]\ngame = \xff\n")

        with pytest.raises(ConfigurationError):
            load_config(cfg_path)

    def test_max_memory_operations_per_turn_default(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [[agents]]
            id = "p1"
            model = "gpt-4o"
        """)
        )
        cfg = load_config(cfg_path)
        assert cfg.run.max_memory_operations_per_turn == 4

    def test_max_memory_operations_per_turn_explicit(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1
            max_memory_operations_per_turn = 8

            [[agents]]
            id = "p1"
            model = "gpt-4o"
        """)
        )
        cfg = load_config(cfg_path)
        assert cfg.run.max_memory_operations_per_turn == 8

    def test_max_memory_operations_per_turn_zero(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1
            max_memory_operations_per_turn = 0

            [[agents]]
            id = "p1"
            model = "gpt-4o"
        """)
        )
        cfg = load_config(cfg_path)
        assert cfg.run.max_memory_operations_per_turn == 0

    @pytest.mark.parametrize("value", ["true", '"4"', "4.0"])
    def test_max_memory_operations_per_turn_rejects_coercion(
        self, tmp_path: Path, value: str
    ) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent(
                f"""\
                [run]
                game = "tiny"
                matches = 1
                max_memory_operations_per_turn = {value}

                [[agents]]
                id = "p1"
                model = "gpt-4o"
            """
            )
        )

        with pytest.raises(ConfigurationError):
            load_config(cfg_path)

    def test_max_memory_operations_per_turn_rejects_negative(
        self, tmp_path: Path
    ) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1
            max_memory_operations_per_turn = -1

            [[agents]]
            id = "p1"
            model = "gpt-4o"
        """)
        )

        with pytest.raises(ConfigurationError):
            load_config(cfg_path)
