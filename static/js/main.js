
document.addEventListener('DOMContentLoaded', function() {
    // Get all navigation items
    const navItems = document.querySelectorAll('.nav .item');
    
    // Function to switch views
    function switchView(viewId) {
        // Hide all views
        document.querySelectorAll('.view').forEach(view => {
            view.classList.remove('active');
        });
        
        // Show the selected view
        document.getElementById(viewId).classList.add('active');
        
        // Update active nav item
        navItems.forEach(item => {
            item.classList.remove('active');
        });
        
        // Find and activate the clicked nav item
        const activeNavItem = document.querySelector(`.nav .item[data-view="${viewId}"]`);
        if (activeNavItem) {
            activeNavItem.classList.add('active');
        }
        if (viewId === 'live-view') {
            renderLiveView();
        }
        if (viewId === 'dashboard') {
            renderDashboard();
        }
        if (viewId === 'evacuation') {
            renderEvacuation();
        }
    }
    
    // Add click event listeners to nav items
    navItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            const viewId = this.getAttribute('data-view');
            switchView(viewId);
        });
    });
    
    // Initialize with dashboard view
    switchView('dashboard');

    // Auto-compose Streaming URL / RTSP URL from form inputs
    const hostEl = document.getElementById('cam-host');
    const portEl = document.getElementById('cam-port');
    const protocolEl = document.getElementById('cam-protocol');
    const userEl = document.getElementById('cam-username');
    const passEl = document.getElementById('cam-password');
    const pathEl = document.getElementById('cam-stream-path');
    const urlEl = document.getElementById('cam-stream-url');

    function updateStreamUrl() {
        if (!urlEl) return;
        const protocolRaw = (protocolEl?.value || 'RTSP').toLowerCase();
        const protocol = protocolRaw.replace(/:$/,'');
        const host = (hostEl?.value || '').trim();
        const port = (portEl?.value || '').trim();
        const user = (userEl?.value || '').trim();
        const pass = (passEl?.value || '').trim();
        const streamPathRaw = (pathEl?.value || '').trim().replace(/^\//,'');

        if (!host) { urlEl.value = ''; return; }

        // URL encode the password ONLY for display in the URL preview
        // The password itself is stored in plain text, encoding is only for URL display
        const encodedPass = pass ? encodeURIComponent(pass) : '';
        const authPart = user ? (encodedPass ? `${user}:${encodedPass}@` : `${user}@`) : '';
        const portPart = port ? `:${port}` : '';
        const pathPart = streamPathRaw ? `/${streamPathRaw}` : '';

        urlEl.value = `${protocol}://${authPart}${host}${portPart}${pathPart}`;
    }

    [hostEl, portEl, protocolEl, userEl, passEl, pathEl].forEach(el => {
        if (!el) return;
        el.addEventListener('input', updateStreamUrl);
        el.addEventListener('change', updateStreamUrl);
    });
    updateStreamUrl();

    // Add Camera view: toggle list vs form
    const cameraList = document.getElementById('camera-list');
    const cameraForm = document.getElementById('camera-form');
    const addBtn = document.getElementById('btn-add-camera');
    const cancelBtn = document.getElementById('btn-cancel-camera');
    const listContainer = document.getElementById('camera-list');

    function showCameraList() {
        if (cameraList) cameraList.style.display = 'block';
        if (cameraForm) cameraForm.style.display = 'none';
        // Stop any preview worker when leaving the form
        if (currentEditId) {
            fetch(`/api/cameras/${encodeURIComponent(currentEditId)}/preview/stop`, { method: 'POST' }).catch(()=>{});
        }
    }

    function showCameraForm() {
        if (cameraList) cameraList.style.display = 'none';
        if (cameraForm) cameraForm.style.display = 'block';
        setupFormPreview(null); // reset until populated
    }

    function clearCameraForm() {
        const nameInput = document.querySelector('#camera-form input[placeholder="e.g., Lobby Entrance"]');
        if (nameInput) nameInput.value = '';
        const idInput = document.querySelector('#camera-form input[placeholder="Unique internal identifier"]');
        if (idInput) idInput.value = '';
        const locInput = document.querySelector('#camera-form input[placeholder="e.g., Building A, Floor 2"]');
        if (locInput) locInput.value = '';
        const desc = document.querySelector('#camera-form textarea');
        if (desc) desc.value = '';
        if (hostEl) hostEl.value = '';
        if (portEl) portEl.value = '';
        if (protocolEl) protocolEl.value = 'RTSP';
        if (pathEl) pathEl.value = '';
        if (userEl) userEl.value = '';
        if (passEl) passEl.value = '';
        const macInput = document.querySelector('#camera-form input[placeholder^="e.g., AA:BB"]');
        if (macInput) macInput.value = '';
        const modeSel = document.getElementById('cam-mode');
        if (modeSel) modeSel.value = 'Recognize';
        updateStreamUrl();
    }

    if (addBtn) {
        addBtn.addEventListener('click', function() {
            currentEditId = null;
            clearCameraForm();
            showCameraForm();
        });
    }
    if (cancelBtn) {
        cancelBtn.addEventListener('click', function() {
            showCameraList();
        });
    }
    
    // Search Camera functionality
    const searchBtn = document.getElementById('btn-search-camera');
    if (searchBtn) {
        searchBtn.addEventListener('click', async function() {
            searchBtn.disabled = true;
            searchBtn.textContent = 'Searching...';
            try {
                const res = await fetch('/api/discover');
                if (!res.ok) throw new Error('Failed to discover cameras');
                const devices = await res.json();
                discoveredCameras = devices.filter(d => d.is_camera);
                await renderCameraList();
                if (discoveredCameras.length === 0) {
                    alert('No cameras found on the network.');
                }
            } catch (e) {
                alert('Failed to discover cameras: ' + (e.message || 'Network error'));
            } finally {
                searchBtn.disabled = false;
                searchBtn.textContent = 'Search Camera';
            }
        });
    }
    
    function addDiscoveredCamera(ip, mac, vendor) {
        // Pre-fill the form with discovered camera data
        currentEditId = null;
        clearCameraForm();
        
        // Auto-fill IP and MAC
        if (hostEl) hostEl.value = ip;
        const macInput = document.querySelector('#camera-form input[placeholder^="e.g., AA:BB"]');
        if (macInput) macInput.value = mac;
        
        // Set default name from vendor
        const nameInput = document.querySelector('#camera-form input[placeholder="e.g., Lobby Entrance"]');
        if (nameInput) nameInput.value = (vendor || 'Camera').trim() + ' - ' + ip;
        
        // Set default protocol (most cameras use RTSP)
        if (protocolEl) protocolEl.value = 'RTSP';
        
        // Set default stream path for common cameras
        const pathInput = document.querySelector('#camera-form input[placeholder*="stream"]');
        if (pathInput && vendor) {
            const vendorLower = vendor.toLowerCase();
            if (vendorLower.includes('axis')) {
                pathInput.value = '/axis-media/media.amp';
            } else if (vendorLower.includes('hikvision')) {
                pathInput.value = '/Streaming/Channels/101';
            }
        }
        
        updateStreamUrl();
        showCameraForm();
    }
    
    // Store discovered cameras
    let discoveredCameras = [];
    
    let currentEditId = null;
    function populateCameraForm(cam) {
        const nameInput = document.querySelector('#camera-form input[placeholder="e.g., Lobby Entrance"]');
        if (nameInput) nameInput.value = cam.name || '';
        const idInput = document.querySelector('#camera-form input[placeholder="Unique internal identifier"]');
        if (idInput) idInput.value = cam.idCode || '';
        const locInput = document.querySelector('#camera-form input[placeholder="e.g., Building A, Floor 2"]');
        if (locInput) locInput.value = cam.location || '';
        const desc = document.querySelector('#camera-form textarea');
        if (desc) desc.value = cam.description || '';
        if (hostEl) hostEl.value = cam.ipOrHost || '';
        if (portEl) portEl.value = cam.port != null ? String(cam.port) : '';
        if (protocolEl) protocolEl.value = (cam.protocol || 'rtsp').toUpperCase();
        if (pathEl) pathEl.value = cam.streamPath || '';
        if (userEl) userEl.value = cam.username || '';
        // Get password from either 'password' (plain) or 'passwordEnc' (for backward compatibility)
        if (passEl) passEl.value = cam.password || cam.passwordEnc || '';
        const macInput = document.querySelector('#camera-form input[placeholder^="e.g., AA:BB"]');
        if (macInput) macInput.value = cam.macAddress || '';
        const modeSel = document.getElementById('cam-mode');
        if (modeSel) modeSel.value = cam.mode || 'Recognize';
        const twEnabled = document.getElementById('tw-enabled');
        const twx1 = document.getElementById('tw-x1');
        const twy1 = document.getElementById('tw-y1');
        const twx2 = document.getElementById('tw-x2');
        const twy2 = document.getElementById('tw-y2');
        if (twEnabled) twEnabled.checked = !!cam.lineEnabled;
        if (twx1) twx1.value = cam.lineX1 ?? '';
        if (twy1) twy1.value = cam.lineY1 ?? '';
        if (twx2) twx2.value = cam.lineX2 ?? '';
        if (twy2) twy2.value = cam.lineY2 ?? '';
        const twdir = document.getElementById('tw-direction');
        if (twdir) twdir.value = cam.lineDirection || 'AtoBIsIn';
        const twcm = document.getElementById('tw-countmode');
        if (twcm) twcm.value = cam.lineCountMode || 'BOTH';
        updateStreamUrl();
        setupFormPreview(cam);
        // Start preview worker thread for this camera
        if (cam && cam.idCode) {
            fetch(`/api/cameras/${encodeURIComponent(cam.idCode)}/preview/start`, { method: 'POST' }).catch(()=>{});
        }
        // Wire live update handlers for tripwire controls
        wireTripwireLiveHandlers();
    }

    if (listContainer) {
        listContainer.addEventListener('click', function(e) {
            const target = e.target;
            if (target && target.classList && target.classList.contains('edit-camera-btn')) {
                const id = target.getAttribute('data-id');
                if (id) {
                    const cam = (window.camerasCache || camerasCache || []).find(c => c.idCode === id);
                    if (cam) {
                        currentEditId = id;
                        populateCameraForm(cam);
                    }
                }
                showCameraForm();
                return;
            }
            if (target && target.classList && target.classList.contains('delete-camera-btn')) {
                const id = target.getAttribute('data-id');
                if (!id) return;
                if (!confirm('Delete this camera permanently?')) return;
                fetch(`/api/cameras/${encodeURIComponent(id)}`, { method: 'DELETE' })
                    .then(async (res) => {
                        if (res.ok) {
                            camerasCache = [];
                            await new Promise(resolve => setTimeout(resolve, 100));
                            await renderCameraList(true);
                            renderDashboard();
                            renderLiveView();
                        } else {
                            alert('Failed to delete camera');
                        }
                    })
                    .catch(() => alert('Network error while deleting camera'));
            }
        });

        // Handle active toggle changes
        listContainer.addEventListener('change', function(e) {
            const t = e.target;
            if (t && t.classList && t.classList.contains('cam-active-toggle')) {
                const id = t.getAttribute('data-id');
                const isActive = !!t.checked;
                fetch(`/api/cameras/${encodeURIComponent(id)}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ active: isActive })
                }).then(async (res) => {
                    if (res.ok) {
                        await renderCameraList();
                        renderDashboard();
                        renderLiveView();
                    } else {
                        alert('Failed to update camera state');
                    }
                }).catch(() => alert('Network error while updating camera'));
            }
        });
    }
    // Default: show list
    showCameraList();

    // Dashboard Full View toggle
    const dashFullBtn = document.getElementById('btn-dashboard-full');
    const dashExitBtn = document.getElementById('btn-dashboard-exit');
    function enterDashboardFullView() {
        switchView('dashboard');
        document.body.classList.add('full-dashboard');
        if (dashFullBtn) dashFullBtn.style.display = 'none';
        if (dashExitBtn) dashExitBtn.style.display = 'inline-block';
    }
    function exitDashboardFullView() {
        document.body.classList.remove('full-dashboard');
        if (dashFullBtn) dashFullBtn.style.display = 'inline-block';
        if (dashExitBtn) dashExitBtn.style.display = 'none';
    }
    if (dashFullBtn) dashFullBtn.addEventListener('click', enterDashboardFullView);
    if (dashExitBtn) dashExitBtn.addEventListener('click', exitDashboardFullView);

    // Date filter functionality
    const dateFilter = document.getElementById('date-filter');
    const btnClearDate = document.getElementById('btn-clear-date');
    
    // Helper function to get today's date in YYYY-MM-DD format
    function getTodayDate() {
        const today = new Date();
        const year = today.getFullYear();
        const month = String(today.getMonth() + 1).padStart(2, '0');
        const day = String(today.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    }
    
    // Set default date to today
    if (dateFilter) {
        dateFilter.value = getTodayDate();
    }
    
    async function updateDashboardWithDateFilter() {
        // Default to today's date if no date is selected
        let selectedDate = dateFilter ? dateFilter.value : null;
        if (!selectedDate) {
            selectedDate = getTodayDate();
            if (dateFilter) {
                dateFilter.value = selectedDate;
            }
        }
        
        // Refresh the entire dashboard (including chart) with the selected date
        await renderDashboard();
    }
    
    if (dateFilter) {
        dateFilter.addEventListener('change', updateDashboardWithDateFilter);
    }
    
    if (btnClearDate) {
        btnClearDate.addEventListener('click', function() {
            if (dateFilter) {
                // Set to today's date instead of clearing
                dateFilter.value = getTodayDate();
                updateDashboardWithDateFilter();
            }
        });
    }

    // Notification system
    function showNotification(message, type = 'success') {
        // Remove existing notification if any
        const existing = document.getElementById('camera-notification');
        if (existing) existing.remove();
        
        const notification = document.createElement('div');
        notification.id = 'camera-notification';
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? '#28a745' : '#dc3545'};
            color: white;
            padding: 16px 24px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 10000;
            font-size: 14px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 8px;
            animation: slideIn 0.3s ease-out;
        `;
        notification.innerHTML = `
            <span>${type === 'success' ? '✓' : '✗'}</span>
            <span>${message}</span>
        `;
        document.body.appendChild(notification);
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-out';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
    
    // Add CSS animation for notification
    if (!document.getElementById('notification-styles')) {
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            @keyframes slideIn {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
            @keyframes slideOut {
                from {
                    transform: translateX(0);
                    opacity: 1;
                }
                to {
                    transform: translateX(100%);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }

    // API helpers
    let camerasCache = [];
    async function fetchCameras(forceRefresh = false) {
        try {
            // Add cache-busting parameter if force refresh
            const url = forceRefresh ? `/api/cameras?t=${Date.now()}` : '/api/cameras';
            const res = await fetch(url, {
                cache: 'no-store',
                headers: {
                    'Cache-Control': 'no-cache'
                }
            });
            if (!res.ok) return [];
            camerasCache = await res.json();
            return camerasCache;
        } catch (_) { return []; }
    }

    function cameraListItemHtml(cam) {
        const id = cam.idCode || '';
        const name = cam.name || '';
        const host = cam.ipOrHost || '';
        const active = cam.active ? 'checked' : '';
        return `
        <div style="display:flex; align-items:center; justify-content: space-between;">
            <div>
                <div style="font-weight:600;">${name}</div>
                <div style="font-size:12px; opacity:.7;">ID: ${id} • ${host}</div>
            </div>
            <div style="display:flex; gap:12px; align-items:center;">
                <label class="switch" aria-label="Toggle camera ${name}">
                    <input type="checkbox" class="cam-active-toggle" data-id="${id}" ${active}>
                    <span class="track"></span>
                    <span class="thumb"></span>
                </label>
                <button class="edit-camera-btn" data-id="${id}" style="background: rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.1); padding: 6px 12px; border-radius: 6px;">Edit</button>
                <button class="delete-camera-btn" data-id="${id}" style="background: #fff; color: var(--warning-red); border: 1px solid rgba(220,53,69,0.4); padding: 6px 12px; border-radius: 6px;">Delete</button>
            </div>
        </div>`;
    }
    
    function discoveredCameraListItemHtml(cam) {
        const vendor = cam.vendor || 'Unknown Vendor';
        const ip = cam.ip_address || '';
        const mac = cam.mac_address || '';
        const status = cam.status || 'can_add';
        const existingId = cam.existing_camera_id || '';
        const isAdded = status === 'already_added';
        
        let statusBadge = '';
        if (isAdded) {
            statusBadge = `<span style="padding: 4px 8px; border-radius: 4px; background: #28a74520; color: #28a745; font-size: 11px; font-weight: 600;">Already Added</span>`;
        }
        
        return `
        <div style="display:flex; align-items:center; justify-content: space-between; opacity: ${isAdded ? '0.7' : '1'};">
            <div>
                <div style="font-weight:600;">${vendor}</div>
                <div style="font-size:12px; opacity:.7;">IP: ${ip}${mac ? ' • MAC: ' + mac : ''}${existingId ? ' • ID: ' + existingId : ''}</div>
            </div>
            <div style="display:flex; gap:12px; align-items:center;">
                ${statusBadge}
                ${!isAdded ? `<button class="btn-add-discovered" data-ip="${ip}" data-mac="${mac}" data-vendor="${vendor}" style="background: #007bff; color: white; border: none; padding: 6px 16px; border-radius: 6px; cursor: pointer; font-size: 14px;">Add</button>` : ''}
            </div>
        </div>`;
    }

    async function renderCameraList(forceRefresh = false) {
        const list = document.getElementById('camera-list');
        if (!list) return;
        const container = list.querySelector('div');
        if (!container) return;
        const cams = await fetchCameras(forceRefresh);
        
        let html = '';
        
        // Render existing cameras
        if (cams.length > 0) {
            html += cams.map(cameraListItemHtml).join('');
        }
        
        // Render discovered cameras (if any)
        if (discoveredCameras.length > 0) {
            if (cams.length > 0) {
                html += '<div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(0,0,0,0.1);">';
            }
            html += '<div style="font-weight: 600; margin-bottom: 12px; color: #666; font-size: 14px;">Discovered Cameras</div>';
            html += discoveredCameras.map(discoveredCameraListItemHtml).join('');
            if (cams.length > 0) {
                html += '</div>';
            }
        }
        
        // Show message if no cameras added
        if (html === '' && discoveredCameras.length === 0) {
            html = `
                <div style="text-align: center; padding: 40px 20px; color: rgba(0,0,0,0.5);">
                    <div style="font-size: 48px; margin-bottom: 16px;">📷</div>
                    <div style="font-size: 18px; font-weight: 600; margin-bottom: 8px;">No cameras added yet</div>
                    <div style="font-size: 14px; margin-bottom: 20px;">Click "Add Camera" to get started</div>
                </div>
            `;
        }
        
        container.innerHTML = html;
        
        // Attach event listeners for add buttons
        container.querySelectorAll('.btn-add-discovered').forEach(btn => {
            btn.addEventListener('click', function() {
                const ip = this.getAttribute('data-ip');
                const mac = this.getAttribute('data-mac');
                const vendor = this.getAttribute('data-vendor');
                addDiscoveredCamera(ip, mac, vendor);
            });
        });
    }

    async function renderLiveView() {
        const container = document.getElementById('live-cards');
        if (!container) return;
        const cams = await fetchCameras();
        // Check if we should show webcam test tile
        let showWebcam = false;
        try {
            const cfgRes = await fetch('/api/config');
            if (cfgRes.ok) {
                const cfg = await cfgRes.json();
                showWebcam = !!cfg.showwebcam;
            }
        } catch(_) {}

        const tiles = [];
        if (showWebcam) {
            tiles.push(`
                <article class="card" style="grid-column: span 6;">
                    <div style="display:flex; flex-direction:column; gap: 12px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div style="font-weight:600;">Webcam (Test)</div>
                            <div style="font-size:12px; opacity:.7;">Recognize</div>
                        </div>
                        <img src="/api/stream/WEBCAM" alt="Webcam" style="width:100%; height: 320px; object-fit: cover; border-radius: 8px; background:#000;"/>
                    </div>
                </article>
            `);
        }

        tiles.push(...cams.map(c => {
            const isActive = !!c.active;
            const img = isActive ? `
                <div class="live-wrap" data-id="${encodeURIComponent(c.idCode)}" style="position:relative; width:100%; height: 320px;">
                    <img class="live-img" src="/api/stream/${encodeURIComponent(c.idCode)}" alt="${c.name}" style="width:100%; height: 100%; object-fit: cover; border-radius: 8px; background:#000;"/>
                    <canvas class="live-canvas" style="position:absolute; inset:0; border-radius:8px; pointer-events:none;"></canvas>
                    <div class="live-tools" style="position:absolute; right:8px; top:8px; display:flex; gap:6px;">
                        <button class="btn-tw-set" title="Set Tripwire" style="background: rgba(0,0,0,0.5); color:#fff; border:0; padding:6px 8px; border-radius:6px;">Set Tripwire</button>
                        <button class="btn-tw-save" title="Save" style="display:none; background: var(--primary-blue); color:#fff; border:0; padding:6px 8px; border-radius:6px;">Save</button>
                        <button class="btn-tw-cancel" title="Cancel" style="display:none; background: rgba(0,0,0,0.5); color:#fff; border:0; padding:6px 8px; border-radius:6px;">Cancel</button>
                    </div>
                </div>
            ` : `<div style="height:320px; background:#333; color:#fff; display:grid; place-items:center; border-radius:8px;">Offline</div>`;
            return `
                <article class="card" style="grid-column: span 6;">
                    <div style="display:flex; flex-direction:column; gap: 12px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div style="font-weight:600;">${c.name || c.idCode}</div>
                            <div style="font-size:12px; opacity:.7;">${c.mode || ''}</div>
                        </div>
                        ${img}
                    </div>
                </article>
            `;
        }));

        if (!tiles.length) {
            container.innerHTML = '<div class="card" style="grid-column: span 12;">No cameras configured.</div>';
        } else {
            container.innerHTML = tiles.join('');
            setupTripwireDrawing(container);
        }
    }

    // Chart instance storage
    let peopleFlowChart = null;

    function processPresenceDataForChart(presence, selectedDate = null) {
        let startDate, endDate;
        
        if (selectedDate) {
            // Use the selected date (00:00 to 23:59 of that date in local timezone)
            // Parse the date string (YYYY-MM-DD) and create date in local timezone
            const [year, month, day] = selectedDate.split('-').map(Number);
            startDate = new Date(year, month - 1, day, 0, 0, 0, 0);
            endDate = new Date(year, month - 1, day, 23, 59, 59, 999);
        } else {
            // Default to last 24 hours from now
            const now = new Date();
            endDate = now;
            startDate = new Date(now.getTime() - 24 * 60 * 60 * 1000);
        }
        
        // Create hourly buckets for the selected date or last 24 hours
        const hourlyData = {};
        
        if (selectedDate) {
            // Create 24 hourly buckets for the selected date (00:00 to 23:00)
            // Use UTC to match event timestamps from API
            const [year, month, day] = selectedDate.split('-').map(Number);
            for (let hour = 0; hour < 24; hour++) {
                // Create date in UTC to match API timestamp format
                const hourDate = new Date(Date.UTC(year, month - 1, day, hour, 0, 0, 0));
                const hourKey = hourDate.toISOString().substring(0, 13) + ':00:00';
                hourlyData[hourKey] = { enter: 0, exit: 0 };
            }
        } else {
            // Create hourly buckets for last 24 hours
            for (let i = 23; i >= 0; i--) {
                const hourTime = new Date(endDate.getTime() - i * 60 * 60 * 1000);
                const hourKey = hourTime.toISOString().substring(0, 13) + ':00:00';
                hourlyData[hourKey] = { enter: 0, exit: 0 };
            }
        }
        
        // Process presence events
        // Note: API already filters by date, so all events here are for the selected date
        presence.forEach(ev => {
            try {
                const eventTime = new Date(ev.timestamp);
                
                // Round to nearest hour in UTC (to match bucket keys)
                const hourTime = new Date(eventTime);
                hourTime.setUTCMinutes(0, 0, 0);
                const hourKey = hourTime.toISOString().substring(0, 13) + ':00:00';
                
                // Match to hourly bucket (both keys are in UTC, so they should match)
                if (hourlyData[hourKey]) {
                    if (ev.event === 'enter') {
                        hourlyData[hourKey].enter++;
                    } else if (ev.event === 'exit') {
                        hourlyData[hourKey].exit++;
                    }
                }
            } catch (e) {
                // Skip invalid timestamps
                console.error('Error processing event:', e, ev);
            }
        });
        
        // Convert to arrays for chart
        const labels = [];
        const enterData = [];
        const exitData = [];
        
        Object.keys(hourlyData).sort().forEach(key => {
            const hourTime = new Date(key);
            const hourLabel = hourTime.toLocaleString('en-US', { 
                month: 'short', 
                day: 'numeric', 
                hour: 'numeric',
                hour12: true 
            });
            labels.push(hourLabel);
            enterData.push(hourlyData[key].enter);
            exitData.push(hourlyData[key].exit);
        });
        
        return { labels, enterData, exitData };
    }

    async function renderDashboard() {
        console.log('🔵 [DEBUG] renderDashboard() called');
        const cams = await fetchCameras();
        console.log('🔵 [DEBUG] Fetched cameras:', cams.length);
        
        // Get the selected date (defaults to today)
        const selectedDate = dateFilter ? (dateFilter.value || getTodayDate()) : getTodayDate();
        const isToday = selectedDate === getTodayDate();
        
        // Fetch live counts for camera list and totals
        let countsData = {};
        let globalInTotal = 0;
        let globalOutTotal = 0;
        let remaining = 0;
        try {
            const countsRes = await fetch('/api/counts');
            if (countsRes.ok) {
                countsData = await countsRes.json();
                console.log('🔵 [DEBUG] Fetched counts from API:', countsData);
                
                // Calculate totals from live counts
                for (const camId in countsData) {
                    const camCounts = countsData[camId];
                    globalInTotal += camCounts.count_in || 0;
                    globalOutTotal += camCounts.count_out || 0;
                }
                remaining = Math.max(0, globalInTotal - globalOutTotal);
            }
        } catch (e) {
            console.error('🔵 [DEBUG] Error fetching counts:', e);
        }
        
        // If viewing a past date (not today), use date-filtered stats instead
        if (!isToday) {
            try {
                const statsRes = await fetch(`/api/count-log/stats?date=${selectedDate}`);
                if (statsRes.ok) {
                    const stats = await statsRes.json();
                    globalInTotal = stats.total_in || 0;
                    globalOutTotal = stats.total_out || 0;
                    remaining = stats.remaining || 0;
                    console.log('🔵 [DEBUG] Fetched date-filtered stats for past date:', stats);
                }
            } catch (e) {
                console.error('🔵 [DEBUG] Error fetching date-filtered stats:', e);
            }
        }
        
        // Build countsByCam from API data (for camera list display)
        const countsByCam = {};
        for (const camId in countsData) {
            const camCounts = countsData[camId];
            countsByCam[camId] = {
                IN: camCounts.count_in || 0,
                OUT: camCounts.count_out || 0
            };
        }
        
        const total = cams.length;
        const active = cams.filter(c => !!c.active).length;
        const inactive = total - active;
        const totalEl = document.getElementById('dash-total');
        const activeEl = document.getElementById('dash-active');
        const inactiveEl = document.getElementById('dash-inactive');
        if (totalEl) totalEl.textContent = String(total);
        if (activeEl) activeEl.textContent = String(active);
        if (inactiveEl) inactiveEl.textContent = String(inactive);

        // Update totals from date-filtered stats
        const inTotEl = document.getElementById('dash-in-total');
        const outTotEl = document.getElementById('dash-out-total');
        const remEl = document.getElementById('dash-remaining');
        if (inTotEl) {
            inTotEl.textContent = String(globalInTotal);
            console.log('🔵 [DEBUG] Updated dash-in-total to:', globalInTotal);
        }
        if (outTotEl) {
            outTotEl.textContent = String(globalOutTotal);
            console.log('🔵 [DEBUG] Updated dash-out-total to:', globalOutTotal);
        }
        if (remEl) {
            remEl.textContent = String(remaining);
            console.log('🔵 [DEBUG] Updated dash-remaining to:', remaining);
        }

        // Render people flow chart (use date-filtered data)
        const chartCanvas = document.getElementById('people-flow-chart');
        if (chartCanvas && typeof Chart !== 'undefined') {
            console.log('🔵 [DEBUG] Rendering chart for date:', selectedDate);
            const presence = await fetchPresence(selectedDate); // Fetch date-filtered presence
            console.log('🔵 [DEBUG] Fetched presence events:', presence.length, 'for date:', selectedDate);
            const chartData = processPresenceDataForChart(presence, selectedDate);
            console.log('🔵 [DEBUG] Chart data processed:', chartData.labels.length, 'hours, IN:', chartData.enterData, 'OUT:', chartData.exitData);
            
            // Update chart title based on selected date
            const chartTitleEl = document.getElementById('chart-title');
            if (chartTitleEl) {
                const dateObj = new Date(selectedDate);
                const dateStr = dateObj.toLocaleDateString('en-US', { 
                    month: 'short', 
                    day: 'numeric', 
                    year: 'numeric' 
                });
                const isToday = selectedDate === getTodayDate();
                chartTitleEl.textContent = isToday 
                    ? `People Flow Trend (Today - ${dateStr})`
                    : `People Flow Trend (${dateStr})`;
            }
            
            if (peopleFlowChart) {
                peopleFlowChart.destroy();
                peopleFlowChart = null;
            }
            
            peopleFlowChart = new Chart(chartCanvas, {
                type: 'line',
                data: {
                    labels: chartData.labels,
                    datasets: [
                        {
                            label: 'IN',
                            data: chartData.enterData,
                            borderColor: '#2072BB',
                            backgroundColor: 'rgba(32, 114, 187, 0.1)',
                            tension: 0.4,
                            fill: true,
                            borderWidth: 2,
                        },
                        {
                            label: 'OUT',
                            data: chartData.exitData,
                            borderColor: '#E39763',
                            backgroundColor: 'rgba(227, 151, 99, 0.1)',
                            tension: 0.4,
                            fill: true,
                            borderWidth: 2,
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    layout: {
                        padding: {
                            top: 5,
                            bottom: 5,
                            left: 5,
                            right: 5
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            padding: 8,
                            titleFont: {
                                size: 12
                            },
                            bodyFont: {
                                size: 11
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: {
                                display: false
                            },
                            ticks: {
                                maxRotation: 45,
                                minRotation: 45,
                                font: {
                                    size: 9
                                },
                                padding: 4
                            }
                        },
                        y: {
                            beginAtZero: true,
                            ticks: {
                                stepSize: 1,
                                font: {
                                    size: 9
                                },
                                padding: 4
                            },
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            }
                        }
                    }
                }
            });
        }

        const listEl = document.getElementById('dash-camera-list');
        if (listEl) {
            if (!cams.length) {
                listEl.innerHTML = '<div style="opacity:.7;">No cameras configured.</div>';
            } else {
                listEl.innerHTML = cams.map(c => {
                    const statusColor = c.active ? 'var(--success-green)' : 'var(--warning-red)';
                    const statusText = c.active ? 'Active' : 'Inactive';
                    const subtitle = `${c.idCode || ''} • ${c.ipOrHost || ''} • ${c.mode || ''}`;
                    const cnt = countsByCam[c.idCode] || { IN: 0, OUT: 0 };
                    return `
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div style="display:flex; flex-direction:column; gap:2px;">
                                <span style="font-weight:600;">${c.name || c.idCode}</span>
                                <span style="font-size:12px; opacity:.7;">${subtitle}</span>
                            </div>
                            <div style="display:flex; align-items:center; gap:12px;">
                                <span style="background: rgba(32,114,187,0.08); padding:2px 8px; border-radius:999px; font-size:12px;">IN: <b>${cnt.IN}</b></span>
                                <span style="background: rgba(227,151,99,0.15); padding:2px 8px; border-radius:999px; font-size:12px;">OUT: <b>${cnt.OUT}</b></span>
                                <span style="color:${statusColor}; font-weight:600;">${statusText}</span>
                            </div>
                        </div>
                    `;
                }).join('');
            }
        }
    }

    // Save camera
    const saveBtn = document.getElementById('btn-save-camera');
    if (saveBtn) {
        saveBtn.addEventListener('click', async function() {
            const payload = {
                name: (document.querySelector('#camera-form input[placeholder="e.g., Lobby Entrance"]')?.value || '').trim(),
                location: (document.querySelector('#camera-form input[placeholder="e.g., Building A, Floor 2"]')?.value || '').trim(),
                description: (document.querySelector('#camera-form textarea')?.value || '').trim(),
                ipOrHost: hostEl?.value || '',
                port: portEl?.value || '',
                protocol: protocolEl?.value?.toLowerCase() || 'rtsp',
                streamPath: pathEl?.value || '',
                username: userEl?.value || '',
                password: passEl?.value || '',
                macAddress: (document.querySelector('#camera-form input[placeholder^="e.g., AA:BB"]')?.value || '').trim(),
                mode: (document.getElementById('cam-mode')?.value || 'Recognize'),
                active: true,
                lineEnabled: !!document.getElementById('tw-enabled')?.checked,
                lineX1: Number(document.getElementById('tw-x1')?.value || '') || null,
                lineY1: Number(document.getElementById('tw-y1')?.value || '') || null,
                lineX2: Number(document.getElementById('tw-x2')?.value || '') || null,
                lineY2: Number(document.getElementById('tw-y2')?.value || '') || null,
                lineDirection: (document.getElementById('tw-direction')?.value || 'AtoBIsIn'),
                lineCountMode: (document.getElementById('tw-countmode')?.value || 'BOTH')
            };
            const isEditing = !!currentEditId;
            try {
                const endpoint = isEditing ? `/api/cameras/${encodeURIComponent(currentEditId)}` : '/api/cameras';
                if (isEditing) payload.idCode = currentEditId;
                const res = await fetch(endpoint, {
                    method: isEditing ? 'PUT' : 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (res.ok) {
                    const cameraData = await res.json().catch(() => ({}));
                    const cameraName = cameraData.name || payload.name || 'Camera';
                    
                    // Show success notification
                    showNotification(`${isEditing ? 'Updated' : 'Added'} camera "${cameraName}" successfully!`, 'success');
                    
                    // Clear form
                    clearCameraForm();
                    currentEditId = null;
                    
                    // Clear cache and force refresh
                    camerasCache = [];
                    // Refresh discovered cameras to update statuses
                    if (discoveredCameras.length > 0) {
                        try {
                            const discoverRes = await fetch('/api/discover');
                            if (discoverRes.ok) {
                                const devices = await discoverRes.json();
                                discoveredCameras = devices.filter(d => d.is_camera);
                            }
                        } catch (_) {}
                    }
                    // Force refresh camera list with a small delay to ensure config is written
                    await new Promise(resolve => setTimeout(resolve, 100));
                    await renderCameraList(true);
                    
                    // Switch to camera list view
                    showCameraList();
                    
                    // Refresh dashboard and live view
                    renderDashboard();
                    renderLiveView();
                } else {
                    const err = await res.json().catch(() => ({}));
                    showNotification('Failed to save camera' + (err.error ? `: ${err.error}` : ''), 'error');
                }
            } catch (e) {
                showNotification('Network error while saving camera', 'error');
            }
        });
    }

    // Initial list render from API if available
    renderCameraList();

    // Presence rendering
    async function fetchPresence(date = null) {
        try {
            let url = '/api/presence';
            if (date) {
                url += `?date=${date}`;
            }
            const res = await fetch(url);
            if (!res.ok) return [];
            const data = await res.json();
            return Array.isArray(data) ? data : [];
        } catch (_) { return []; }
    }
    function renderPresenceTable(rows) {
        const tbody = document.getElementById('presence-tbody');
        if (!tbody) return;
        tbody.innerHTML = rows.slice().reverse().map((r) => {
            const ts = r.timestamp || '';
            const nm = r.name || `Person_${r.person_id ?? ''}`;
            const ev = r.event || '';
            // Show thumbnail image instead of "View" link
            const img = r.image_path ? 
                `<a href="/${r.image_path}" target="_blank" style="display:inline-block;">
                    <img src="/${r.image_path}" alt="Person ${r.person_id}" 
                         style="width:60px; height:60px; object-fit:cover; border-radius:6px; cursor:pointer; border:1px solid rgba(0,0,0,0.1);"
                         onerror="this.style.display='none'; this.parentElement.innerHTML='No image';"/>
                 </a>` : 
                '<span style="color:rgba(0,0,0,0.4); font-size:12px;">No image</span>';
            return `<tr>
                <td style="padding:8px; border-top:1px solid rgba(0,0,0,0.06);">${ts}</td>
                <td style="padding:8px; border-top:1px solid rgba(0,0,0,0.06);">${nm}</td>
                <td style="padding:8px; border-top:1px solid rgba(0,0,0,0.06);">${ev}</td>
                <td style="padding:8px; border-top:1px solid rgba(0,0,0,0.06);">${img}</td>
            </tr>`;
        }).join('');
        const total = rows.length;
        const ins = rows.filter(r => r.event === 'enter').length;
        const outs = rows.filter(r => r.event === 'exit').length;
        const elIn = document.getElementById('presence-in');
        const elOut = document.getElementById('presence-out');
        const elTot = document.getElementById('presence-total');
        if (elIn) elIn.textContent = String(ins);
        if (elOut) elOut.textContent = String(outs);
        if (elTot) elTot.textContent = String(total);
    }
    async function renderPresence() {
        const rows = await fetchPresence();
        renderPresenceTable(rows);
    }
    const btnRefreshPresence = document.getElementById('btn-refresh-presence');
    if (btnRefreshPresence) {
        btnRefreshPresence.addEventListener('click', renderPresence);
    }
    // Fetch and display peak hours
    async function renderPeakHours() {
        try {
            const res = await fetch('/api/peak-hours');
            if (res.ok) {
                const data = await res.json();
                const entryEl = document.getElementById('peak-entry-time');
                const exitEl = document.getElementById('peak-exit-time');
                if (entryEl) entryEl.textContent = data.peak_entry_time || '--';
                if (exitEl) exitEl.textContent = data.peak_exit_time || '--';
            }
        } catch (e) {
            console.error('Error fetching peak hours:', e);
        }
    }

    // Fetch and display camera performance
    async function renderCameraPerformance() {
        try {
            const res = await fetch('/api/camera-performance');
            if (res.ok) {
                const data = await res.json();
                const listEl = document.getElementById('camera-performance-list');
                if (listEl) {
                    if (data.length === 0) {
                        listEl.innerHTML = '<div style="display: flex; justify-content: space-between;"><span>No cameras</span><span style="color: rgba(0,0,0,0.3);">--</span></div>';
                    } else {
                        listEl.innerHTML = data.map(cam => {
                            const color = cam.status === 'good' ? 'var(--success-green)' : 
                                          cam.status === 'warning' ? 'var(--warning-orange)' : 
                                          'var(--warning-red)';
                            return `
                                <div style="display: flex; justify-content: space-between;">
                                    <span>${cam.name}</span>
                                    <span style="color: ${color}; font-weight:600;">${cam.performance}%</span>
                                </div>
                            `;
                        }).join('');
                    }
                }
            }
        } catch (e) {
            console.error('Error fetching camera performance:', e);
        }
    }

    // Refresh all reports data
    async function refreshReportsData() {
        await Promise.all([
            renderPresence(),
            renderPeakHours(),
            renderCameraPerformance()
        ]);
    }

    // Auto refresh when Reports tab is shown
    const reportsNav = document.querySelector('.nav .item[data-view="reports"]');
    if (reportsNav) {
        reportsNav.addEventListener('click', () => setTimeout(refreshReportsData, 50));
    }
    // Auto-refresh presence events every 3 seconds when Reports view is active
    let presenceInterval = null;
    const reportsView = document.getElementById('reports');
    
    function startPresenceAutoRefresh() {
        if (!presenceInterval && reportsView && reportsView.classList.contains('active')) {
            presenceInterval = setInterval(refreshReportsData, 3000);
        }
    }
    
    function stopPresenceAutoRefresh() {
        if (presenceInterval) {
            clearInterval(presenceInterval);
            presenceInterval = null;
        }
    }
    
    if (reportsView) {
        // Watch for class changes (active/inactive)
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.attributeName === 'class') {
                    const isActive = reportsView.classList.contains('active');
                    if (isActive) {
                        // View became active - start auto-refresh
                        startPresenceAutoRefresh();
                        refreshReportsData(); // Immediate refresh
                    } else {
                        // View became inactive - stop auto-refresh
                        stopPresenceAutoRefresh();
                    }
                }
            });
        });
        observer.observe(reportsView, { attributes: true, attributeFilter: ['class'] });
    }
    
    // Also handle tab clicks directly
    if (reportsNav) {
        reportsNav.addEventListener('click', () => {
            setTimeout(() => {
                startPresenceAutoRefresh();
            }, 100);
        });
    }
    
    // Check initial state
    if (reportsView && reportsView.classList.contains('active')) {
        startPresenceAutoRefresh();
    }
    // Initial render for dashboard and presence
    renderPresence();

    // People rename UI
    async function fetchPeople() {
        try {
            const res = await fetch('/api/people');
            if (!res.ok) return [];
            return await res.json();
        } catch(_) { return []; }
    }
    function peopleRowHtml(p) {
        const pid = p.person_id;
        const nm = p.name || `Person_${pid}`;
        const imgs = p.images || 0;
        return `
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="min-width:80px; font-weight:600;">#${pid}</div>
                <input data-pid="${pid}" class="person-name-input" type="text" value="${nm.replace(/"/g,'&quot;')}" style="flex:1; padding:8px; border:1px solid rgba(0,0,0,0.1); border-radius:8px;"/>
                <button data-pid="${pid}" class="person-save-btn" style="background: var(--primary-blue); color:#fff; border:0; padding:6px 10px; border-radius:6px;">Save</button>
                <div style="opacity:.7; font-size:12px;">${imgs} images</div>
            </div>
        `;
    }
    async function renderPeople() {
        const list = document.getElementById('people-list');
        if (!list) return;
        const people = await fetchPeople();
        if (!people.length) { list.innerHTML = '<div style="opacity:.7;">No people yet.</div>'; return; }
        list.innerHTML = people.map(peopleRowHtml).join('');
    }
    const btnRefreshPeople = document.getElementById('btn-refresh-people');
    if (btnRefreshPeople) btnRefreshPeople.addEventListener('click', renderPeople);
    // Save handler
    document.addEventListener('click', async (e) => {
        const t = e.target;
        if (t && t.classList && t.classList.contains('person-save-btn')) {
            const pid = t.getAttribute('data-pid');
            const input = document.querySelector(`.person-name-input[data-pid="${pid}"]`);
            const name = (input?.value || '').trim();
            if (!name) { alert('Name required'); return; }
            try {
                const res = await fetch(`/api/people/${encodeURIComponent(pid)}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name })
                });
                if (!res.ok) throw new Error('Failed');
                await renderPeople();
            } catch(_) { alert('Failed to save name'); }
        }
    });
    // Periodically refresh dashboard counts
    setInterval(() => { renderDashboard(); }, 5000);
    
    // Auto-refresh evacuation data when view is active
    let evacuationRefreshInterval = null;
    const originalSwitchView = switchView;
    switchView = function(viewId) {
        originalSwitchView(viewId);
        // Clear existing interval
        if (evacuationRefreshInterval) {
            clearInterval(evacuationRefreshInterval);
            evacuationRefreshInterval = null;
        }
        // Start auto-refresh if evacuation view is active
        if (viewId === 'evacuation') {
            evacuationRefreshInterval = setInterval(() => {
                renderEvacuation();
            }, 3000); // Refresh every 3 seconds
        }
    };

    // Evacuation functions
    async function renderEvacuation() {
        // Update people count from presence data
        const presence = await fetchPresence();
        const remaining = Math.max(0, 
            presence.filter(e => e.event === 'enter').length - 
            presence.filter(e => e.event === 'exit').length
        );
        const peopleCountEl = document.getElementById('evac-people-count');
        if (peopleCountEl) {
            peopleCountEl.textContent = String(remaining);
        }
        
        // Update last updated timestamp
        const lastUpdateEl = document.getElementById('evac-last-update');
        if (lastUpdateEl) {
            lastUpdateEl.textContent = new Date().toLocaleString();
        }
        
        // Render people table
        await renderEvacuationPeople();
    }

    // Render evacuation people table with thumbnails and last seen times
    async function renderEvacuationPeople() {
        const tbody = document.getElementById('evacuation-people-tbody');
        const jsonDataEl = document.getElementById('evacuation-json-data');
        if (!tbody) return;

        try {
            // Fetch people and presence data
            const [people, presence] = await Promise.all([
                fetchPeople(),
                fetchPresence()
            ]);

            // Build a map of person_id to last seen time and location
            const personLastSeen = {};
            const personLocations = {};
            presence.forEach(event => {
                const pid = event.person_id;
                if (pid) {
                    const eventTime = new Date(event.timestamp);
                    const existing = personLastSeen[pid];
                    if (!existing || eventTime > existing.time) {
                        personLastSeen[pid] = {
                            time: eventTime,
                            timestamp: event.timestamp,
                            camera_id: event.camera_id || 'Unknown',
                            event: event.event
                        };
                    }
                    // Track location from camera_id
                    if (event.camera_id) {
                        personLocations[pid] = event.camera_id;
                    }
                }
            });

            // Build people data array with images
            const peopleData = await Promise.all(people.map(async (person) => {
                const pid = person.person_id;
                const lastSeen = personLastSeen[pid];
                
                // Get first image from person folder
                let thumbnail = '';
                if (person.images > 0) {
                    try {
                        // Try to get first image from person_images folder
                        const folderPath = `person_images/person_${pid}`;
                        // Since we can't list directory via API, we'll use the image_path from presence events
                        const personEvents = presence.filter(e => e.person_id === pid && e.image_path);
                        if (personEvents.length > 0) {
                            // Get the most recent image
                            const latestEvent = personEvents.reduce((latest, current) => {
                                return new Date(current.timestamp) > new Date(latest.timestamp) ? current : latest;
                            }, personEvents[0]);
                            thumbnail = latestEvent.image_path || '';
                        }
                    } catch (e) {
                        console.error('Error loading thumbnail:', e);
                    }
                }

                return {
                    person_id: pid,
                    name: person.name || `Person_${pid}`,
                    thumbnail: thumbnail,
                    last_seen: lastSeen ? lastSeen.timestamp : null,
                    last_seen_time: lastSeen ? lastSeen.time : null,
                    location: lastSeen ? lastSeen.camera_id : 'Unknown',
                    status: lastSeen ? (lastSeen.event === 'enter' ? 'Inside' : 'Outside') : 'Unknown',
                    images_count: person.images || 0
                };
            }));

            // Sort by last seen time (most recent first)
            peopleData.sort((a, b) => {
                if (!a.last_seen_time && !b.last_seen_time) return 0;
                if (!a.last_seen_time) return 1;
                if (!b.last_seen_time) return -1;
                return b.last_seen_time - a.last_seen_time;
            });

            // Render table
            if (peopleData.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" style="padding: 24px; text-align: center; opacity: 0.7;">No people data available</td></tr>';
            } else {
                tbody.innerHTML = peopleData.map(p => {
                    const thumbnailHtml = p.thumbnail ? 
                        `<img src="/${p.thumbnail.replace(/\\/g, '/')}" alt="${p.name}" style="width: 50px; height: 50px; object-fit: cover; border-radius: 6px; border: 2px solid rgba(0,0,0,0.1);" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'50\' height=\'50\'%3E%3Crect width=\'50\' height=\'50\' fill=\'%23ddd\'/%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\' dy=\'.3em\' fill=\'%23999\' font-size=\'12\'%3ENo Image%3C/text%3E%3C/svg%3E';" />` :
                        `<div style="width: 50px; height: 50px; background: rgba(0,0,0,0.1); border-radius: 6px; display: grid; place-items: center; font-size: 10px; color: rgba(0,0,0,0.5);">No Photo</div>`;
                    
                    const lastSeenDisplay = p.last_seen ? 
                        new Date(p.last_seen).toLocaleString() : 
                        '<span style="opacity: 0.5;">Never</span>';
                    
                    const statusColor = p.status === 'Inside' ? 'var(--warning-red)' : 
                                        p.status === 'Outside' ? 'var(--success-green)' : 
                                        'rgba(0,0,0,0.5)';
                    
                    return `
                        <tr style="border-bottom: 1px solid rgba(0,0,0,0.06);">
                            <td style="padding: 12px;">${thumbnailHtml}</td>
                            <td style="padding: 12px; font-weight: 600;">#${p.person_id}</td>
                            <td style="padding: 12px;">${p.name}</td>
                            <td style="padding: 12px; font-size: 13px;">${lastSeenDisplay}</td>
                            <td style="padding: 12px; font-size: 13px;">${p.location}</td>
                            <td style="padding: 12px;">
                                <span style="padding: 4px 10px; background: ${statusColor === 'var(--warning-red)' ? 'rgba(220,53,69,0.1)' : statusColor === 'var(--success-green)' ? 'rgba(40,167,69,0.1)' : 'rgba(0,0,0,0.05)'}; color: ${statusColor}; border-radius: 999px; font-size: 12px; font-weight: 600;">${p.status}</span>
                            </td>
                        </tr>
                    `;
                }).join('');
            }

            // Update JSON view data
            if (jsonDataEl) {
                jsonDataEl.textContent = JSON.stringify(peopleData, null, 2);
            }
        } catch (error) {
            console.error('Error rendering evacuation people:', error);
            tbody.innerHTML = '<tr><td colspan="6" style="padding: 24px; text-align: center; color: var(--warning-red);">Error loading people data</td></tr>';
        }
    }

    // Evacuation button handlers
    const btnTriggerEvac = document.getElementById('btn-trigger-evacuation');
    const btnResetEvac = document.getElementById('btn-reset-evacuation');
    const btnRefreshEvacPeople = document.getElementById('btn-refresh-evac-people');
    const btnToggleJsonView = document.getElementById('btn-toggle-json-view');
    
    if (btnTriggerEvac) {
        btnTriggerEvac.addEventListener('click', function() {
            if (confirm('🚨 Trigger evacuation alert? This will notify all systems.')) {
                // Update status to active evacuation
                const statusEl = document.querySelector('#evacuation-status > div');
                if (statusEl) {
                    statusEl.style.background = 'rgba(220,53,69,0.1)';
                    statusEl.style.borderLeftColor = 'var(--warning-red)';
                    statusEl.querySelector('div:last-child').textContent = 'Evacuation Active';
                    statusEl.querySelector('div:last-child').style.color = 'var(--warning-red)';
                }
                alert('🚨 Evacuation alert triggered!');
            }
        });
    }
    
    if (btnResetEvac) {
        btnResetEvac.addEventListener('click', function() {
            const statusEl = document.querySelector('#evacuation-status > div');
            if (statusEl) {
                statusEl.style.background = 'rgba(40,167,69,0.1)';
                statusEl.style.borderLeftColor = 'var(--success-green)';
                statusEl.querySelector('div:last-child').textContent = 'Normal';
                statusEl.querySelector('div:last-child').style.color = 'var(--success-green)';
            }
            renderEvacuation();
        });
    }

    if (btnRefreshEvacPeople) {
        btnRefreshEvacPeople.addEventListener('click', async function() {
            await renderEvacuationPeople();
        });
    }

    // JSON view toggle
    if (btnToggleJsonView) {
        let jsonViewActive = false;
        btnToggleJsonView.addEventListener('click', function() {
            jsonViewActive = !jsonViewActive;
            const tableView = document.getElementById('evacuation-people-table');
            const jsonView = document.getElementById('evacuation-json-view');
            if (tableView && jsonView) {
                if (jsonViewActive) {
                    tableView.style.display = 'none';
                    jsonView.style.display = 'block';
                    btnToggleJsonView.textContent = '📋 Table View';
                } else {
                    tableView.style.display = 'block';
                    jsonView.style.display = 'none';
                    btnToggleJsonView.textContent = '📄 JSON View';
                }
            }
        });
    }

    // Tripwire drawing handlers
    function setupTripwireDrawing(root) {
        const wraps = root.querySelectorAll('.live-wrap');
        wraps.forEach(wrap => {
            const id = wrap.getAttribute('data-id');
            if (!id) return;
            const img = wrap.querySelector('.live-img');
            const canvas = wrap.querySelector('.live-canvas');
            const btnSet = wrap.querySelector('.btn-tw-set');
            const btnSave = wrap.querySelector('.btn-tw-save');
            const btnCancel = wrap.querySelector('.btn-tw-cancel');
            if (!img || !canvas || !btnSet || !btnSave || !btnCancel) return;
            const ctx = canvas.getContext('2d');
            let drawing = false;
            let p1 = null;
            let p2 = null;

            function resizeCanvas() {
                // Match canvas size to image client box
                const rect = wrap.getBoundingClientRect();
                canvas.width = rect.width;
                canvas.height = rect.height;
                redraw();
            }
            function clearCanvas() {
                ctx.clearRect(0,0,canvas.width, canvas.height);
            }
            function redraw() {
                clearCanvas();
                if (p1 && p2) {
                    ctx.strokeStyle = 'rgba(255,165,0,0.9)';
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.stroke();
                }
            }
            function enableDrawUI(enable) {
                canvas.style.pointerEvents = enable ? 'auto' : 'none';
                btnSave.style.display = enable ? 'inline-block' : 'none';
                btnCancel.style.display = enable ? 'inline-block' : 'none';
                // When enabling, ensure canvas sized
                if (enable) {
                    resizeCanvas();
                } else {
                    p1 = p2 = null;
                    redraw();
                }
            }
            function toCanvasPoint(evt) {
                const r = canvas.getBoundingClientRect();
                const x = evt.clientX - r.left;
                const y = evt.clientY - r.top;
                return { x, y };
            }
            function toImageCoords(pt) {
                const cw = canvas.width, ch = canvas.height;
                const iw = img.naturalWidth || img.clientWidth;
                const ih = img.naturalHeight || img.clientHeight;
                const scaleX = iw / cw;
                const scaleY = ih / ch;
                return { x: Math.round(pt.x * scaleX), y: Math.round(pt.y * scaleY) };
            }

            let dragStart = null;
            function onDown(e) {
                drawing = true;
                dragStart = toCanvasPoint(e);
                p1 = dragStart;
                p2 = dragStart;
                redraw();
                e.preventDefault();
            }
            function onMove(e) {
                if (!drawing) return;
                p2 = toCanvasPoint(e);
                redraw();
                e.preventDefault();
            }
            function onUp(e) {
                if (!drawing) return;
                drawing = false;
                p2 = toCanvasPoint(e);
                redraw();
                e.preventDefault();
            }

            let ro = new ResizeObserver(resizeCanvas);
            ro.observe(wrap);

            btnSet.addEventListener('click', () => {
                enableDrawUI(true);
            });
            btnCancel.addEventListener('click', () => {
                enableDrawUI(false);
            });
            btnSave.addEventListener('click', async () => {
                if (!p1 || !p2) { alert('Draw a line first.'); return; }
                const a = toImageCoords(p1);
                const b = toImageCoords(p2);
                // Save to backend
                try {
                    const res = await fetch(`/api/cameras/${encodeURIComponent(id)}`, {
                        method: 'PUT',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ lineEnabled: true, lineX1: a.x, lineY1: a.y, lineX2: b.x, lineY2: b.y })
                    });
                    if (!res.ok) throw new Error('Failed');
                    enableDrawUI(false);
                } catch (e) {
                    alert('Failed to save tripwire');
                }
            });

            // Pointer events
            canvas.addEventListener('pointerdown', onDown);
            canvas.addEventListener('pointermove', onMove);
            window.addEventListener('pointerup', onUp);
        });
    }

    // Live-update tripwire settings to backend on change so preview reflects immediately
    function wireTripwireLiveHandlers() {
        const dirEl = document.getElementById('tw-direction');
        const cmEl = document.getElementById('tw-countmode');
        const enEl = document.getElementById('tw-enabled');
        const x1El = document.getElementById('tw-x1');
        const y1El = document.getElementById('tw-y1');
        const x2El = document.getElementById('tw-x2');
        const y2El = document.getElementById('tw-y2');
        async function pushUpdate(partial) {
            if (!currentEditId) return;
            try {
                await fetch(`/api/cameras/${encodeURIComponent(currentEditId)}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(partial)
                });
            } catch(_) {}
        }
        if (dirEl) dirEl.onchange = () => pushUpdate({ lineDirection: dirEl.value });
        if (cmEl) cmEl.onchange = () => pushUpdate({ lineCountMode: cmEl.value });
        if (enEl) enEl.onchange = () => pushUpdate({ lineEnabled: !!enEl.checked });
        function coordsPayload() {
            const x1 = Number(x1El?.value || '');
            const y1 = Number(y1El?.value || '');
            const x2 = Number(x2El?.value || '');
            const y2 = Number(y2El?.value || '');
            return {
                lineX1: isFinite(x1) && x1 > 0 ? x1 : null,
                lineY1: isFinite(y1) && y1 > 0 ? y1 : null,
                lineX2: isFinite(x2) && x2 > 0 ? x2 : null,
                lineY2: isFinite(y2) && y2 > 0 ? y2 : null,
            };
        }
        const onCoordChange = () => pushUpdate(coordsPayload());
        if (x1El) x1El.onchange = onCoordChange;
        if (y1El) y1El.onchange = onCoordChange;
        if (x2El) x2El.onchange = onCoordChange;
        if (y2El) y2El.onchange = onCoordChange;
    }

    // Form embedded preview with tripwire drawing
    function setupFormPreview(cam) {
        const wrap = document.getElementById('cam-preview-wrap');
        const img = document.getElementById('cam-preview-img');
        const canvas = document.getElementById('cam-preview-canvas');
        const btnSet = document.getElementById('btn-form-tw-set');
        const btnSave = document.getElementById('btn-form-tw-save');
        const btnCancel = document.getElementById('btn-form-tw-cancel');
        const note = document.getElementById('cam-preview-note');
        if (!wrap || !img || !canvas || !btnSet || !btnSave || !btnCancel) return;
        let ctx = canvas.getContext('2d');
        // Default hidden
        wrap.style.display = 'none';
        img.src = '';
        // Helper to enable with a given stream id
        function enableWithStream(streamId, showNote) {
            wrap.style.display = 'flex';
            img.src = `/api/stream/${encodeURIComponent(streamId)}`;
            if (note) note.style.display = showNote ? 'block' : 'none';
        }
        // Prefer the actual camera stream if editing an existing camera
        if (cam && cam.idCode) {
            enableWithStream(cam.idCode, !cam.active);
        } else {
            // Fallback to webcam preview if enabled in config
            fetch('/api/config').then(r => r.ok ? r.json() : { showwebcam: false }).then(cfg => {
                if (cfg && cfg.showwebcam) {
                    enableWithStream('WEBCAM', true);
                }
            }).catch(() => {});
            // No camera yet; keep hidden until saved or webcam available
            return;
        }

        let p1 = null, p2 = null, drawing = false;
        function resizeCanvas() {
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
            redraw();
        }
        function clearCanvas() { ctx.clearRect(0,0,canvas.width,canvas.height); }
        function redraw() {
            clearCanvas();
            if (p1 && p2) {
                ctx.strokeStyle = 'rgba(255,165,0,0.9)';
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.moveTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
                ctx.stroke();
            }
        }
        function enableDraw(en) {
            canvas.style.pointerEvents = en ? 'auto' : 'none';
            btnSave.style.display = en ? 'inline-block' : 'none';
            btnCancel.style.display = en ? 'inline-block' : 'none';
            if (en) resizeCanvas(); else { p1 = p2 = null; redraw(); }
        }
        function toCanvasPoint(evt) {
            const r = canvas.getBoundingClientRect();
            return { x: evt.clientX - r.left, y: evt.clientY - r.top };
        }
        function toImageCoords(pt) {
            const cw = canvas.width, ch = canvas.height;
            const iw = img.naturalWidth || img.clientWidth;
            const ih = img.naturalHeight || img.clientHeight;
            const scaleX = iw / cw;
            const scaleY = ih / ch;
            return { x: Math.round(pt.x * scaleX), y: Math.round(pt.y * scaleY) };
        }
        function onDown(e){ drawing = true; p1 = toCanvasPoint(e); p2 = p1; redraw(); e.preventDefault(); }
        function onMove(e){ if(!drawing) return; p2 = toCanvasPoint(e); redraw(); e.preventDefault(); }
        function onUp(e){ if(!drawing) return; drawing = false; p2 = toCanvasPoint(e); redraw(); e.preventDefault(); }
        let ro = new ResizeObserver(resizeCanvas); ro.observe(canvas.parentElement);
        btnSet.onclick = () => enableDraw(true);
        btnCancel.onclick = () => enableDraw(false);
        btnSave.onclick = async () => {
            if (!p1 || !p2) { alert('Draw a line first.'); return; }
            const a = toImageCoords(p1); const b = toImageCoords(p2);
            try {
                const res = await fetch(`/api/cameras/${encodeURIComponent(cam.idCode)}`, {
                    method: 'PUT', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ lineEnabled: true, lineX1: a.x, lineY1: a.y, lineX2: b.x, lineY2: b.y })
                });
                if (!res.ok) throw new Error('failed');
                enableDraw(false);
            } catch(_) { alert('Failed to save tripwire'); }
        };
        canvas.addEventListener('pointerdown', onDown);
        canvas.addEventListener('pointermove', onMove);
        window.addEventListener('pointerup', onUp);
    }
});
