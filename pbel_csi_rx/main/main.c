#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#include "esp_event.h"
#include "esp_log.h"
#include "esp_now.h"
#include "esp_wifi.h"
#include "nvs_flash.h"

#define WIFI_CHANNEL       6
#define CSI_MAX_BYTES      256
#define CSI_QUEUE_LENGTH   16
#define CSI_PRINT_EVERY    5

static const char *TAG = "PBEL_RX";

/* Transmitter Wi-Fi MAC confirmed earlier on COM3. */
static const uint8_t TX_MAC[6] = {
    0xa4, 0xf0, 0x0f, 0x69, 0x1a, 0x94
};

typedef struct {
    uint8_t mac[6];
    uint32_t timestamp_us;
    int8_t rssi;
    uint8_t channel;
    uint8_t sig_mode;
    uint16_t len;
    bool first_word_invalid;
    int8_t iq[CSI_MAX_BYTES];
} csi_frame_t;

static QueueHandle_t csi_queue;
static volatile uint32_t csi_received;
static volatile uint32_t espnow_received;
static volatile uint32_t csi_queue_dropped;
static volatile uint32_t csi_too_large;

static bool from_transmitter(const uint8_t *mac)
{
    return mac != NULL && memcmp(mac, TX_MAC, sizeof(TX_MAC)) == 0;
}

/* Called in the Wi-Fi task: count and copy only; do not print here. */
static void on_espnow_receive(
    const esp_now_recv_info_t *info,
    const uint8_t *data,
    int len)
{
    if (info != NULL && data != NULL && len > 0 &&
        from_transmitter(info->src_addr)) {
        espnow_received++;
    }
}

/* CSI memory is valid only while this callback is running. */
static void on_csi_receive(void *context, wifi_csi_info_t *info)
{
    (void)context;

    if (info == NULL || info->buf == NULL ||
        !from_transmitter(info->mac)) {
        return;
    }

    csi_received++;

    /* Keep serial output below the default 115200 baud capacity. */
    if ((csi_received % CSI_PRINT_EVERY) != 0) {
        return;
    }

    if (info->len == 0 || info->len > CSI_MAX_BYTES) {
        csi_too_large++;
        return;
    }

    csi_frame_t frame = {0};
    memcpy(frame.mac, info->mac, sizeof(frame.mac));
    frame.timestamp_us = info->rx_ctrl.timestamp;
    frame.rssi = info->rx_ctrl.rssi;
    frame.channel = info->rx_ctrl.channel;
    frame.sig_mode = info->rx_ctrl.sig_mode;
    frame.len = info->len;
    frame.first_word_invalid = info->first_word_invalid;
    memcpy(frame.iq, info->buf, frame.len);

    if (xQueueSend(csi_queue, &frame, 0) != pdTRUE) {
        csi_queue_dropped++;
    }
}

/* Runs separately so serial printing does not hold up Wi-Fi reception. */
static void csi_output_task(void *argument)
{
    (void)argument;
    csi_frame_t frame;
    char hex[CSI_MAX_BYTES * 2 + 1];
    static const char digits[] = "0123456789abcdef";

    while (true) {
        if (xQueueReceive(csi_queue, &frame, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        for (unsigned i = 0; i < frame.len; i++) {
            uint8_t value = (uint8_t)frame.iq[i];
            hex[2 * i] = digits[value >> 4];
            hex[2 * i + 1] = digits[value & 0x0f];
        }
        hex[2 * frame.len] = '\0';

        printf("CSI_DATA,%lu,"
               "%02x:%02x:%02x:%02x:%02x:%02x,"
               "%d,%u,%u,%u,%u,%s\n",
               (unsigned long)frame.timestamp_us,
               frame.mac[0], frame.mac[1], frame.mac[2],
               frame.mac[3], frame.mac[4], frame.mac[5],
               (int)frame.rssi,
               (unsigned)frame.channel,
               (unsigned)frame.sig_mode,
               (unsigned)frame.len,
               (unsigned)frame.first_word_invalid,
               hex);
    }
}

void app_main(void)
{
    esp_err_t nvs_result = nvs_flash_init();
    if (nvs_result == ESP_ERR_NVS_NO_FREE_PAGES ||
        nvs_result == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        nvs_result = nvs_flash_init();
    }
    ESP_ERROR_CHECK(nvs_result);

    csi_queue = xQueueCreate(CSI_QUEUE_LENGTH, sizeof(csi_frame_t));
    configASSERT(csi_queue != NULL);
    configASSERT(xTaskCreate(csi_output_task, "csi_output",
                             4096, NULL, 3, NULL) == pdPASS);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    wifi_init_config_t wifi_config = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&wifi_config));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW20));
    ESP_ERROR_CHECK(
        esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE));

    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_register_recv_cb(on_espnow_receive));

    /* A disconnected station needs promiscuous reception for CSI. */
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = false,
        .stbc_htltf2_en = false,
        .ltf_merge_en = false,
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = 0
    };

    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(on_csi_receive, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    uint8_t receiver_mac[6];
    ESP_ERROR_CHECK(esp_wifi_get_mac(WIFI_IF_STA, receiver_mac));
    ESP_LOGI(TAG,
             "RX_READY channel=%d bandwidth=HT20 mac=%02x:%02x:%02x:%02x:%02x:%02x",
             WIFI_CHANNEL,
             receiver_mac[0], receiver_mac[1], receiver_mac[2],
             receiver_mac[3], receiver_mac[4], receiver_mac[5]);

    while (true) {
        vTaskDelay(pdMS_TO_TICKS(5000));
        ESP_LOGI(TAG,
                 "RX_STATUS espnow=%lu csi=%lu queue_dropped=%lu too_large=%lu",
                 (unsigned long)espnow_received,
                 (unsigned long)csi_received,
                 (unsigned long)csi_queue_dropped,
                 (unsigned long)csi_too_large);
    }
}