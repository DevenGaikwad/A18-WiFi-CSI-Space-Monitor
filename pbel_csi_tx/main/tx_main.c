#include <inttypes.h>
#include <stdint.h>
#include <string.h>

#include "esp_err.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"

#define WIFI_CHANNEL 6
#define TX_INTERVAL_MS 20  // Nominally 50 packets per second

static const char *TAG = "CSI_TX";
static const uint8_t broadcast_mac[6] = {
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};

void app_main(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES ||
        err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    wifi_init_config_t wifi_cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&wifi_cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20));
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(
        esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE)
    );

    ESP_ERROR_CHECK(esp_now_init());

    esp_now_peer_info_t peer = {0};
    memcpy(peer.peer_addr, broadcast_mac, sizeof(broadcast_mac));
    peer.channel = WIFI_CHANNEL;
    peer.ifidx = WIFI_IF_STA;
    peer.encrypt = false;
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
        esp_now_rate_config_t rate_config = {
        .phymode = WIFI_PHY_MODE_HT20,
        .rate = WIFI_PHY_RATE_MCS0_LGI,
        .ersu = false,
        .dcm = false
    };
    ESP_ERROR_CHECK(
        esp_now_set_peer_rate_config(broadcast_mac, &rate_config)
    );
    ESP_LOGI(TAG, "TX_RATE HT20 MCS0");

    uint8_t mac[6];
    ESP_ERROR_CHECK(esp_wifi_get_mac(WIFI_IF_STA, mac));

    ESP_LOGI(TAG, "TX_READY channel=%d bandwidth=HT20 interval_ms=%d mac=" MACSTR,
             WIFI_CHANNEL, TX_INTERVAL_MS, MAC2STR(mac));

    uint32_t sequence = 0;
    uint32_t queued = 0;
    uint32_t errors = 0;
    TickType_t last_wake = xTaskGetTickCount();

    while (1) {
        // Constant-length packet. The receiver measures CSI from its radio signal.
        uint32_t packet = sequence++;
        err = esp_now_send(
            broadcast_mac, (const uint8_t *)&packet, sizeof(packet)
        );

        if (err == ESP_OK) {
            queued++;
        } else {
            errors++;
            if (errors <= 3) {
                ESP_LOGW(TAG, "Send queue error: %s", esp_err_to_name(err));
            }
        }

        if (sequence % 250 == 0) {
            ESP_LOGI(TAG, "TX_STATUS queued=%" PRIu32 " errors=%" PRIu32,
                     queued, errors);
        }

        vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(TX_INTERVAL_MS));
    }
}