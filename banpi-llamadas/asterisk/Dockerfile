# Asterisk 20 (paquete de Ubuntu 24.04) con ARI y External Media (chan_rtp).
FROM ubuntu:24.04
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends asterisk asterisk-modules \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /var/spool/asterisk/recording
CMD ["asterisk", "-f", "-vvv"]
