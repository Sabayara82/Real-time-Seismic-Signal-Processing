from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd
import numpy as np
import os

DATA_CLIENT = "EARTHSCOPE"

QUAKE_TIME = UTCDateTime("2004-12-26T00:58:53")
QUAKE_LABEL = "Sumatra_2004"

PRE_QUAKE_SECONDS  = 10000 
POST_QUAKE_SECONDS = 50000 

NETWORK  = "II"
STATION  = "DGAR"
LOCATION = "00"
CHANNEL  = "BH*"   # fetches all three axis data

OUT_DIR = "seismic_output"

def fetch_waveforms(client, network, station, location, channel, t_start, t_end):
    stream = client.get_waveforms(
        network=network,
        station=station,
        location=location,
        channel=channel,
        starttime=t_start,
        endtime=t_end,
    )
    return stream


def remove_instrument_response(stream, client, network, station, location):
    inventory = client.get_stations(
        network=network,
        station=station,
        location=location,
        channel="BH*",
        starttime=stream[0].stats.starttime,
        endtime=stream[0].stats.endtime,
        level="response",
    )

    stream.remove_response(
        inventory=inventory,
        output="VEL",      
        water_level=60,
    )
    return stream


def stream_to_dataframes(stream, quake_time):
    dfs = {}
    for trace in stream:
        channel = trace.stats.channel
        sr      = trace.stats.sampling_rate
        t_start = trace.stats.starttime

        n_samples  = len(trace.data)
        abs_times  = np.array([t_start + i / sr for i in range(n_samples)])
        rel_seconds = np.array([(t - quake_time) for t in abs_times])

        df = pd.DataFrame({
            "utc_time":            [str(t) for t in abs_times],
            "seconds_from_origin": rel_seconds,
            "velocity_m_s":        trace.data,
        })
        dfs[channel] = df
    return dfs


def save_csvs(dfs, out_dir, label):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for channel, df in dfs.items():
        fname = os.path.join(out_dir, f"{label}_{channel}.csv")
        df.to_csv(fname, index=False)
        saved.append(fname)

    if len(dfs) > 1:
        combined = None
        for channel, df in dfs.items():
            col = df[["seconds_from_origin", "velocity_m_s"]].rename(
                columns={"velocity_m_s": f"vel_{channel}_m_s"}
            )
            combined = col if combined is None else pd.merge(
                combined, col, on="seconds_from_origin", how="outer"
            )
        if combined is not None:
            fname_combined = os.path.join(out_dir, f"{label}_combined.csv")
            combined.to_csv(fname_combined, index=False)
            saved.append(fname_combined)
    return saved


def main():

    client  = Client(DATA_CLIENT)
    t_start = QUAKE_TIME - PRE_QUAKE_SECONDS
    t_end   = QUAKE_TIME + POST_QUAKE_SECONDS

    stream = fetch_waveforms(client, NETWORK, STATION, LOCATION, CHANNEL, t_start, t_end)
    stream = remove_instrument_response(stream, client, NETWORK, STATION, LOCATION)

    dfs   = stream_to_dataframes(stream, QUAKE_TIME)
    saved = save_csvs(dfs, OUT_DIR, QUAKE_LABEL)

if __name__ == "__main__":
    main()