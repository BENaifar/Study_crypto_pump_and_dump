"""Telegram history collection (opt-in, never run by tests). [orig: Data/Telegram/get_channel_post.py]"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from pnd.common.io import save_pickle
from pnd.common.seeding import set_global_seed
from pnd.config import Config
from pnd.data.telegram.channels import entity_from_line, read_channel_lines
from pnd.pipeline.registry import register

logger = logging.getLogger(__name__)


class TelethonChannelFetcher:
    """Q36: download every message of every broadcast channel listed in ``Pump_channel2``."""

    def __init__(self, raw_dir: Path, channel_list: Path, session_file: Path,
                 limit: int = 500, total_count_limit: int = 500000, entity_slice_start: int = 13) -> None:
        self.raw_dir = Path(raw_dir)
        self.channel_list = Path(channel_list)
        self.session_file = Path(session_file)
        self.limit = limit
        self.total_count_limit = total_count_limit
        self.entity_slice_start = entity_slice_start

    def run(self) -> None:
        try:
            from telethon import TelegramClient
            from telethon.tl.functions.messages import GetHistoryRequest
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError("Telegram fetching needs the optional extra: uv sync --extra telegram") from exc

        # D2: credentials from the environment instead of hard-coded placeholders
        api_id = int(os.environ["TELEGRAM_API_ID"])
        api_hash = os.environ["TELEGRAM_API_HASH"]
        client = TelegramClient(str(self.session_file), api_id, api_hash)

        async def main():
            Channels = read_channel_lines(self.channel_list)

            failed_channel = []
            # QUIRK(Q36): existing ids = file names with the last 4 chars stripped [orig: get_channel_post.py:L30]
            exist_channel = list(map(lambda y: y[:-4], os.listdir(self.raw_dir)))

            for channel in Channels:

                offset_id = 0
                limit = self.limit
                all_messages = []
                total_messages = 0
                total_count_limit = self.total_count_limit
                # QUIRK(Q36): entity = channel[13:-1] [orig: get_channel_post.py:L40]
                entity = entity_from_line(channel, self.entity_slice_start)

                try:
                    my_channel = await client.get_entity(entity)
                    if not my_channel.broadcast:
                        logger.info("entity: " + entity + " is not broad cast channel")
                        continue

                except:
                    logger.info("entity: " + entity + " can not be use!")
                    failed_channel.append(entity)
                    continue

                if str(my_channel.id) in exist_channel:
                    logger.info("entity: " + entity + " has existed")
                    continue

                while True:
                    logger.info("Current Offset ID is: %s ; Total Messages: %s", offset_id, total_messages)
                    history = await client(GetHistoryRequest(
                        peer=my_channel,
                        offset_id=offset_id,
                        offset_date=None,
                        add_offset=0,
                        limit=limit,
                        max_id=0,
                        min_id=0,
                        hash=0
                    ))
                    if not history.messages:
                        break
                    messages = history.messages
                    for message in messages:
                        all_messages.append(message.to_dict())
                    offset_id = messages[len(messages) - 1].id
                    total_messages = len(all_messages)
                    if total_count_limit != 0 and total_messages >= total_count_limit:
                        break

                logger.info("pause")
                save_pickle(all_messages, self.raw_dir / (str(my_channel.id) + ".pkl"))

        with client:
            client.loop.run_until_complete(main())


@register
class FetchTelegramStage:
    """CLI wrapper around :class:`TelethonChannelFetcher`."""

    name = "fetch-telegram"

    def run(self, cfg: Config) -> None:
        set_global_seed(cfg.seed)
        p = cfg.paths.telegram
        tf_cfg = cfg.telegram_fetch
        TelethonChannelFetcher(
            raw_dir=p.raw_dir,
            channel_list=p.channel_list,
            session_file=p.session_file,
            limit=tf_cfg.limit,
            total_count_limit=tf_cfg.total_count_limit,
            entity_slice_start=tf_cfg.entity_slice_start,
        ).run()
