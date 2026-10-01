/*
 * soapy_capture - record raw IQ from a SoapySDR device (SDRplay RSP1B).
 *
 * A single-channel receive capture written as interleaved int8 IQ ("CS8"),
 * the same on-disk layout as the HackRF captures already read by
 * atsc3lib.frontend.read_hackrf_iq.  The SDRplay driver is reached through
 * the SoapySDR C API (Device.h), not the Python SWIG bindings.
 *
 *   soapy_capture --freq 587e6 --rate 10e6 --bw 8e6 --out file.iq \
 *                 --duration 10 --ifgr 30 --rfgr 4 [--index 0]
 *   soapy_capture --probe
 *
 * Exit 0 on success, non-zero on any setup/stream error.  --probe enumerates
 * the requested driver and exits 0 when a device is present.
 */

#include <SoapySDR/Device.h>
#include <SoapySDR/Formats.h>
#include <SoapySDR/Constants.h>
#include <SoapySDR/Errors.h>
#include <SoapySDR/Types.h>

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define RX_SOAPY_CHANNEL 0u
#define RX_SOAPY_DIRECTION SOAPY_SDR_RX
#define SOAPY_SDRPLAY_DRIVER "sdrplay"
#define SOAPY_RX_ANTENNA "RX"
#define SOAPY_IFGR_GAIN "IFGR"
#define SOAPY_RFGR_GAIN "RFGR"
#define SOAPY_IFGR_DEFAULT 30.0
#define SOAPY_RFGR_DEFAULT 4.0
/* The SDRplay driver advertises CS16 and CF32 only; stream the native CS16.
 * The default output is int16 IQ (``--cs8`` down-converts int16 >> 8 to the
 * int8 layout). */
#define STREAM_FORMAT SOAPY_SDR_CS16
#define READ_TIMEOUT_US 1000000L
#define CS16_ELEM_BYTES 4u
#define CS8_ELEM_BYTES 2u
#define PROBE_DRIVER_ARG "driver=sdrplay"

static volatile sig_atomic_t g_stop = 0;

static void on_signal(int signo) { (void)signo; g_stop = 1; }

static double parse_double(const char *s, const char *what)
{
    char *end = NULL;
    double v = strtod(s, &end);
    if (end == s) {
        fprintf(stderr, "soapy_capture: invalid %s value '%s'\n", what, s);
        exit(2);
    }
    return v;
}

static long long parse_ll(const char *s, const char *what)
{
    char *end = NULL;
    long long v = strtoll(s, &end, 10);
    if (end == s) {
        fprintf(stderr, "soapy_capture: invalid %s value '%s'\n", what, s);
        exit(2);
    }
    return v;
}

static int check_call(int code, const char *what)
{
    if (code == 0) return 0;
    fprintf(stderr, "soapy_capture: %s failed: %s (%d)\n",
            what, SoapySDR_errToStr(code), code);
    return 1;
}

static int probe(void)
{
    size_t len = 0;
    SoapySDRKwargs *results = SoapySDRDevice_enumerateStrArgs(PROBE_DRIVER_ARG, &len);
    for (size_t i = 0; i < len; i++) {
        char *markup = SoapySDRKwargs_toString(&results[i]);
        printf("%s\n", markup ? markup : "?");
        if (markup) SoapySDR_free(markup);
    }
    SoapySDRKwargsList_clear(results, len);
    return len > 0 ? 0 : 1;
}

int main(int argc, char **argv)
{
    const char *out_path = NULL;
    const char *driver = SOAPY_SDRPLAY_DRIVER;
    double freq_hz = 0.0, rate_hz = 10e6, bw_hz = 8e6;
    double ifgr = SOAPY_IFGR_DEFAULT, rfgr = SOAPY_RFGR_DEFAULT;
    long long duration_sec = 10;
    int device_index = 0;
    int do_probe = 0;
    int out_cs8 = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--freq")) freq_hz = parse_double(argv[++i], "--freq");
        else if (!strcmp(a, "--rate")) rate_hz = parse_double(argv[++i], "--rate");
        else if (!strcmp(a, "--bw")) bw_hz = parse_double(argv[++i], "--bw");
        else if (!strcmp(a, "--ifgr")) ifgr = parse_double(argv[++i], "--ifgr");
        else if (!strcmp(a, "--rfgr")) rfgr = parse_double(argv[++i], "--rfgr");
        else if (!strcmp(a, "--duration")) duration_sec = parse_ll(argv[++i], "--duration");
        else if (!strcmp(a, "--out")) out_path = argv[++i];
        else if (!strcmp(a, "--driver")) driver = argv[++i];
        else if (!strcmp(a, "--index")) device_index = (int)parse_ll(argv[++i], "--index");
        else if (!strcmp(a, "--probe")) do_probe = 1;
        else if (!strcmp(a, "--cs8")) out_cs8 = 1;
        else if (!strcmp(a, "--cs16")) out_cs8 = 0;
        else if (!strcmp(a, "--help")) {
            printf("usage: %s --freq <hz> --out <path> [--rate hz] [--bw hz]\n"
                   "        [--duration s] [--ifgr db] [--rfgr n] [--driver sdrplay]\n"
                   "        [--index n] [--cs8] [--probe]\n", argv[0]);
            return 0;
        } else {
            fprintf(stderr, "soapy_capture: unknown argument '%s'\n", a);
            return 2;
        }
    }

    if (do_probe) return probe();

    if (out_path == NULL || freq_hz <= 0.0) {
        fprintf(stderr, "soapy_capture: --freq and --out are required\n");
        return 2;
    }

    size_t len = 0;
    SoapySDRKwargs *results = SoapySDRDevice_enumerateStrArgs(
        "driver=sdrplay", &len);
    if (len == 0 || device_index < 0 || (size_t)device_index >= len) {
        fprintf(stderr, "soapy_capture: no %s device at index %d\n",
                driver, device_index);
        SoapySDRKwargsList_clear(results, len);
        return 1;
    }
    char *args_markup = SoapySDRKwargs_toString(&results[device_index]);
    SoapySDRKwargsList_clear(results, len);
    if (args_markup == NULL) {
        fprintf(stderr, "soapy_capture: failed to read device args\n");
        return 1;
    }

    SoapySDRDevice *device = SoapySDRDevice_makeStrArgs(args_markup);
    SoapySDR_free(args_markup);
    if (device == NULL) {
        fprintf(stderr, "soapy_capture: cannot open device\n");
        return 1;
    }

    int rc = 0;
    rc |= check_call(SoapySDRDevice_setSampleRate(device, RX_SOAPY_DIRECTION,
                    RX_SOAPY_CHANNEL, rate_hz), "setSampleRate");
    rc |= check_call(SoapySDRDevice_setFrequency(device, RX_SOAPY_DIRECTION,
                    RX_SOAPY_CHANNEL, freq_hz, NULL), "setFrequency");
    rc |= check_call(SoapySDRDevice_setBandwidth(device, RX_SOAPY_DIRECTION,
                    RX_SOAPY_CHANNEL, bw_hz), "setBandwidth");
    rc |= check_call(SoapySDRDevice_setAntenna(device, RX_SOAPY_DIRECTION,
                    RX_SOAPY_CHANNEL, SOAPY_RX_ANTENNA), "setAntenna");
    if (SoapySDRDevice_hasGainMode(device, RX_SOAPY_DIRECTION, RX_SOAPY_CHANNEL)) {
        rc |= check_call(SoapySDRDevice_setGainMode(device, RX_SOAPY_DIRECTION,
                        RX_SOAPY_CHANNEL, false), "setGainMode(manual)");
    }
    rc |= check_call(SoapySDRDevice_setGainElement(device, RX_SOAPY_DIRECTION,
                    RX_SOAPY_CHANNEL, SOAPY_IFGR_GAIN, ifgr), "setGain(IFGR)");
    rc |= check_call(SoapySDRDevice_setGainElement(device, RX_SOAPY_DIRECTION,
                    RX_SOAPY_CHANNEL, SOAPY_RFGR_GAIN, rfgr), "setGain(RFGR)");

    SoapySDRStream *stream = SoapySDRDevice_setupStream(
        device, RX_SOAPY_DIRECTION, STREAM_FORMAT, NULL, 0, NULL);
    if (stream == NULL) {
        fprintf(stderr, "soapy_capture: setupStream failed\n");
        SoapySDRDevice_unmake(device);
        return 1;
    }
    rc |= check_call(SoapySDRDevice_activateStream(device, stream, 0, 0, 0),
                     "activateStream");

    FILE *fp = fopen(out_path, "wb");
    if (fp == NULL) {
        fprintf(stderr, "soapy_capture: cannot open '%s'\n", out_path);
        SoapySDRDevice_deactivateStream(device, stream, 0, 0);
        SoapySDRDevice_closeStream(device, stream);
        SoapySDRDevice_unmake(device);
        return 1;
    }

    size_t mtu = SoapySDRDevice_getStreamMTU(device, stream);
    size_t elem_bytes = SoapySDR_formatToSize(STREAM_FORMAT);
    void *buf = malloc(mtu * elem_bytes);
    if (buf == NULL) {
        fprintf(stderr, "soapy_capture: buffer allocation failed\n");
        fclose(fp);
        SoapySDRDevice_deactivateStream(device, stream, 0, 0);
        SoapySDRDevice_closeStream(device, stream);
        SoapySDRDevice_unmake(device);
        return 1;
    }

    /* CS16 -> CS8 conversion scratch (one interleaved int8 per CS16 sample). */
    int8_t *out8 = out_cs8 ? malloc(mtu * CS8_ELEM_BYTES) : NULL;
    if (out_cs8 && out8 == NULL) {
        fprintf(stderr, "soapy_capture: buffer allocation failed\n");
        free(buf);
        fclose(fp);
        SoapySDRDevice_deactivateStream(device, stream, 0, 0);
        SoapySDRDevice_closeStream(device, stream);
        SoapySDRDevice_unmake(device);
        return 1;
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    long long target = (long long)(rate_hz * (double)duration_sec);
    long long got = 0;
    long long overflows = 0;

    while (!g_stop && got < target) {
        int flags = 0;
        long long time_ns = 0;
        void *buffs[1] = { buf };
        int n = SoapySDRDevice_readStream(device, stream, buffs, mtu,
                                          &flags, &time_ns, READ_TIMEOUT_US);
        if (n > 0) {
            size_t bytes;
            if (!out_cs8) {
                /* Native CS16: int16 I, int16 Q, written unchanged. */
                bytes = (size_t)n * CS16_ELEM_BYTES;
                if (fwrite(buf, 1, bytes, fp) != bytes) {
                    fprintf(stderr, "soapy_capture: write error\n");
                    rc = 1;
                    break;
                }
            } else {
                const int16_t *in16 = (const int16_t *)buf;
                int n16 = n * 2;
                for (int j = 0; j < n16; j++) {
                    out8[j] = (int8_t)(in16[j] >> 8);
                }
                bytes = (size_t)n16;
                if (fwrite(out8, 1, bytes, fp) != bytes) {
                    fprintf(stderr, "soapy_capture: write error\n");
                    rc = 1;
                    break;
                }
            }
            got += n;
        } else if (n == SOAPY_SDR_TIMEOUT) {
            continue;
        } else if (n == SOAPY_SDR_OVERFLOW) {
            overflows++;
        } else if (n < 0) {
            fprintf(stderr, "soapy_capture: readStream: %s (%d)\n",
                    SoapySDR_errToStr(n), n);
            rc = 1;
            break;
        }
    }

    free(out8);
    free(buf);
    fclose(fp);
    SoapySDRDevice_deactivateStream(device, stream, 0, 0);
    SoapySDRDevice_closeStream(device, stream);
    SoapySDRDevice_unmake(device);

    if (rc == 0) {
        fprintf(stderr, "soapy_capture: %lld samples (%lld overflows)\n",
                got, overflows);
    }
    return rc;
}
