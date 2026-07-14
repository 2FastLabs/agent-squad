enum OrderCard {
    /// The widget template (`ui://shop/order-card`). It reads `window.WIDGET_DATA` — the render-only
    /// `structuredContent` injected by the host — and exposes `window.render()` so the host can push
    /// fresh data after an `.app`-only tool call (Refresh).
    static let html = """
    <!-- ui://shop/order-card -->
    <style>
      body { margin: 0; background: transparent; }
      .card { font-family: -apple-system, system-ui; border-radius: 14px; padding: 16px;
              background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,.12); }
      h3 { margin: 0 0 8px; }
      .status { color: #0a7; font-weight: 600; margin: 4px 0; }
      button { margin-top: 12px; border: 0; border-radius: 8px; padding: 8px 14px;
               background: #0a7; color: #fff; font-weight: 600; }
    </style>
    <div class="card">
      <h3>Order <span id="id"></span></h3>
      <p class="status" id="status"></p>
      <p>ETA <strong id="eta"></strong> · <span id="carrier"></span></p>
      <button onclick="refresh()">Refresh</button>
    </div>
    <script>
      function render() {
        const d = window.WIDGET_DATA || {};
        document.getElementById('id').textContent      = d.orderId || '';
        document.getElementById('status').textContent  = d.status  || '';
        document.getElementById('eta').textContent     = d.eta     || '';
        document.getElementById('carrier').textContent = d.carrier || '';
      }
      function refresh() {
        window.webkit.messageHandlers.host.postMessage(JSON.stringify({
          tool: 'refresh_order',
          args: { orderId: (window.WIDGET_DATA || {}).orderId }
        }));
      }
      window.render = render;   // host calls this again after pushing fresh data
      render();                 // initial paint from data injected at document start
    </script>
    """
}
