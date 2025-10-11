<!DOCTYPE html>
<html lang="id">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ $title ?? 'Super Admin' }}</title>
    <link rel="icon" type="image/x-icon" href="{{ asset('favicon.ico') }}">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet" />
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons/font/bootstrap-icons.css" rel="stylesheet">

    <style>
        body {
            background-color: #f4f6f9;
            font-family: 'Poppins', sans-serif;
        }

        .sidebar {
            min-width: 250px;
            min-height: 100vh;
            background-color: #fff;
            border-right: 1px solid #e1e1e1;
        }

        .sidebar .nav-link {
            color: #333;
            font-weight: 500;
            padding: 12px 20px;
            border-radius: 8px;
            margin-bottom: 8px;
        }

        .sidebar .nav-link.active,
        .sidebar .nav-link:hover {
            background-color: #0d6efd;
            color: #fff !important;
        }

        .card-stat {
            border: none;
            border-radius: 16px;
            background: linear-gradient(135deg, var(--color), #96c93d);
            color: #fff;
            transition: transform 0.2s ease;
        }

        .card-stat:hover {
            transform: translateY(-3px);
        }

        .mitra-card img {
            width: 100px;
            height: 100px;
            object-fit: cover;
        }

        .topbar {
            background-color: #fff;
            border-bottom: 1px solid #dee2e6;
            padding: 15px 30px;
        }

        .nav-title {
            font-weight: 600;
            font-size: 18px;
        }

        h2,
        h4,
        h6 {
            font-family: 'Segoe UI', 'Roboto', sans-serif;
        }

        .card h6 {
            font-weight: 600;
        }

        .article-content {
            line-height: 1.7;
            font-size: 1rem;
            text-align: justify;
            white-space: pre-wrap;
            /* Tambahkan ini */
        }

        .modal-article-content h3 {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .modal-article-content .text-muted small {
            font-size: 0.9rem;
        }

        .modal-article-content .article-content {
            line-height: 1.7;
            font-size: 1rem;
            text-align: justify;
        }

        .modal-article-content img {
            max-height: 300px;
            object-fit: cover;
            border-radius: 12px;
            margin-bottom: 20px;
            width: 100%;
            max-width: 100%;
        }
    </style>
</head>

<body>

    <div class="d-flex">
        <!-- Sidebar -->
        <div class="sidebar p-4 shadow-sm">
            <h4 class="text-primary mb-4">Super Admin</h4>
            <div class="nav flex-column">
                <a href="{{ route('admin.dashboard') }}"
                    class="nav-link {{ request()->routeIs('admin.dashboard') ? 'active' : '' }}">
                    <i class="bi bi-speedometer2 me-2"></i> Dashboard
                </a>
                <a href="{{ route('admin.sertifikasi') }}"
                    class="nav-link {{ request()->routeIs('admin.sertifikasi') ? 'active' : '' }}">
                    <i class="bi bi-award-fill me-2"></i> Sertifikasi
                </a>
                <a href="{{ route('admin.pelatihan') }}"
                    class="nav-link {{ request()->routeIs('admin.pelatihan') ? 'active' : '' }}">
                    <i class="bi bi-mortarboard-fill me-2"></i> Pelatihan
                </a>
                <a href="{{ route('admin.loker') }}"
                    class="nav-link {{ request()->routeIs('admin.loker') ? 'active' : '' }}">
                    <i class="bi bi-briefcase-fill me-2"></i> Loker
                </a>
                <a href="{{ route('admin.artikel') }}" class="nav-link"
                    class="nav-link {{ request()->routeIs('admin.artikel') ? 'active' : '' }}">
                    <i class="bi bi-file-earmark-text-fill me-2"></i> Artikel
                </a>
                <div class="nav-item">
                    <a class="nav-link d-flex justify-content-between align-items-center {{ request()->is('admin/users*') ? 'active' : '' }}"
                        data-bs-toggle="collapse" href="#submenuUsers" role="button"
                        aria-expanded="{{ request()->is('admin/users*') ? 'true' : 'false' }}"
                        aria-controls="submenuUsers">
                        <span><i class="bi bi-people-fill me-2"></i> Users</span>
                        <i class="bi bi-chevron-down small"></i>
                    </a>
                    <div class="collapse {{ request()->is('admin/users*') ? 'show' : '' }}" id="submenuUsers">
                        <a href="{{ route('admin.usersalumni') }}"
                            class="nav-link ps-4 {{ request()->routeIs('admin.usersalumni') ? 'active' : '' }}">
                            <i class="bi bi-person-lines-fill me-2"></i> Alumni
                        </a>
                        <a href="{{ route('admin.usersmitra') }}"
                            class="nav-link ps-4 {{ request()->routeIs('admin.usersmitra') ? 'active' : '' }}">
                            <i class="bi bi-building me-2"></i> Mitra
                        </a>
                    </div>
                </div>
                <form id="logout-form" action="{{ route('logout') }}" method="POST" class="mt-auto">
                    @csrf
                    <a href="#" class="nav-link d-flex align-items-center text-danger fw-semibold mb-4"
                        onclick="confirmLogout(event)">
                        <i class="bi bi-box-arrow-right me-2"></i> Logout
                    </a>
                </form>
            </div>
            <div class="mt-auto pt-5 text-muted small">
                &copy; {{ date('Y') }} Super Admin
            </div>
        </div>

        <!-- Main Content -->
        <div class="flex-grow-1">
            <!-- Topbar -->
            <div class="topbar d-flex justify-content-between align-items-center bg-primary text-light p-3">
                <span class="nav-title">{{ $title ?? 'Dashboard' }}</span>
                <span class="text-light">Hi, Admin</span>
            </div>

            <!-- Page Content -->
            <main class="container-fluid py-5 px-4">

                <form method="GET" action="" class="mb-4 px-4">
                    <div class="row g-3 mb-3">

                        {{-- Nama penulis --}}
                        <div class="col-md-4">
                            <label class="form-label fw-semibold small text-muted">Cari Nama penulis</label>
                            <input type="text" class="form-control rounded-3 shadow-sm border border-light-subtle"
                                name="search" value="{{ request('nama_penulis') }}" placeholder="Nama penulis">
                        </div>

                        {{-- Bulan dibuat --}}
                        <div class="col-md-4">
                            <label class="form-label fw-semibold small text-muted">Bulan Dibuat</label>
                            <select name="bulan_dibuat" class="form-select">
                                <option value="" disabled {{ request('bulan_dibuat') ? '' : 'selected' }}>Pilih
                                    Bulan
                                </option>
                                @foreach ([
        '01' => 'Januari',
        '02' => 'Februari',
        '03' => 'Maret',
        '04' => 'April',
        '05' => 'Mei',
        '06' => 'Juni',
        '07' => 'Juli',
        '08' => 'Agustus',
        '09' => 'September',
        '10' => 'Oktober',
        '11' => 'November',
        '12' => 'Desember',
    ] as $val => $label)
                                    <option value="{{ $val }}"
                                        {{ request('dibuat') == $val ? 'selected' : '' }}>
                                        {{ $label }}
                                    </option>
                                @endforeach
                            </select>
                        </div>

                        {{-- Tahun dibuat --}}
                        <div class="col-md-4">
                            <label class="form-label fw-semibold small text-muted">Tahun Dibuat</label>
                            <input type="number" class="form-control" name="tahun_dibuat" placeholder="Tahun dibuat"
                                value="{{ request('tahun_dibuat') }}">
                        </div>

                        {{-- Tombol --}}
                        <div class="col d-flex justify-content-end gap-2 align-items-end">
                            <button type="submit" class="btn btn-primary rounded-pill px-4 shadow-sm">
                                <i class="bi bi-search me-1"></i> Terapkan
                            </button>
                            <a href="{{ route('admin.artikel') }}"
                                class="btn btn-secondary rounded-pill px-4 shadow-sm">
                                <i class="bi bi-arrow-clockwise me-1"></i> Reset
                            </a>
                            <button type="button" class="btn btn-success rounded-pill px-4 shadow-sm"
                                data-bs-toggle="modal" data-bs-target="#modalTambah">
                                <i class="bi bi-plus-circle me-1"></i> Tambah
                            </button>
                        </div>

                    </div>
                </form>

                <div class="table-responsive artikel-table-wrapper">
                    <div id="tableLoadingArtikel" class="d-none text-center py-5">
                        <div class="spinner-border text-primary" role="status" style="width: 3rem; height: 3rem;">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                    </div>

                    <table class="table artikel-table table-hover align-middle mb-0" id="artikelTable">
                        <thead class="table-light text-center text-uppercase small">
                            <tr>
                                <th>No</th>
                                <th>Judul</th>
                                <th>Penulis</th>
                                <th>Tanggal</th>
                                <th>Aksi</th>
                            </tr>
                        </thead>
                        <tbody>
                            @forelse ($artikel as $index => $item)
                                <tr class="text-center">
                                    <td>{{ $index + $artikel->firstItem() }}</td>
                                    <td class="text-start">
                                        <div class="fw-semibold text-center">
                                            {{ \Illuminate\Support\Str::words($item->judul, 10, '...') }}</div>
                                    </td>
                                    <td>{{ $item->penulis }}</td>
                                    <td>{{ $item->created_at->format('d M Y') }}</td>
                                    <td>
                                        <div class="dropdown">
                                            <button class="btn btn-sm btn-primary dropdown-toggle" type="button"
                                                data-bs-toggle="dropdown" aria-expanded="false">
                                                Aksi
                                            </button>
                                            <ul class="dropdown-menu dropdown-menu-end" data-bs-auto-close="outside">
                                                <li>
                                                    <button class="dropdown-item" data-judul="{{ $item->judul }}"
                                                        data-penulis="{{ $item->penulis }}"
                                                        data-tanggal="{{ $item->created_at->format('d M Y') }}"
                                                        data-isi="{{ htmlentities($item->isi) }}"
                                                        data-gambar="{{ asset('storage/' . $item->gambar_artikel) }}"
                                                        onclick="openDetailArtikel(this)">
                                                        <i class="bi bi-eye me-2"></i> Lihat
                                                    </button>
                                                </li>
                                                <li>
                                                    <button class="dropdown-item" data-bs-toggle="modal"
                                                        data-bs-target="#editArtikelModal{{ $item->id }}">
                                                        <i class="bi bi-pencil me-2"></i> Edit
                                                    </button>
                                                </li>
                                                <li>
                                                    <button class="dropdown-item text-danger"
                                                        onclick="confirmDelete('{{ $item->id }}')">
                                                        <i class="bi bi-trash me-2"></i> Hapus
                                                    </button>
                                                </li>
                                            </ul>
                                        </div>
                                    </td>
                                </tr>

                                <!-- Modal Edit Artikel -->
                                <div class="modal fade" id="editArtikelModal{{ $item->id }}" tabindex="-1"
                                    aria-labelledby="editArtikelModalLabel{{ $item->id }}" aria-hidden="true">
                                    <div class="modal-dialog modal-lg">
                                        <div class="modal-content rounded-4 border-0 shadow">
                                            <form id="editForm{{ $item->id }}"
                                                action="{{ route('admin.artikel.update', $item->id) }}"
                                                method="POST" enctype="multipart/form-data">
                                                @csrf
                                                @method('PUT')
                                                <div class="modal-header bg-primary text-white rounded-top-4">
                                                    <h5 class="modal-title fw-semibold"
                                                        id="editArtikelModalLabel{{ $item->id }}">Edit Artikel
                                                    </h5>
                                                    <button type="button" class="btn-close btn-close-white"
                                                        data-bs-dismiss="modal" aria-label="Tutup"></button>
                                                </div>
                                                <div class="modal-body">
                                                    <div class="row g-4">
                                                        <!-- Kolom Kiri -->
                                                        <div class="col-md-6">
                                                            <div class="mb-3">
                                                                <label for="judul{{ $item->id }}"
                                                                    class="form-label fw-semibold">Judul
                                                                    Artikel</label>
                                                                <input type="text" class="form-control shadow-sm"
                                                                    id="judul{{ $item->id }}" name="judul"
                                                                    value="{{ $item->judul }}" required>
                                                            </div>

                                                            <div class="mb-3">
                                                                <label for="penulis{{ $item->id }}"
                                                                    class="form-label fw-semibold">Penulis
                                                                    Artikel</label>
                                                                <input type="text" class="form-control shadow-sm"
                                                                    id="penulis{{ $item->id }}" name="penulis"
                                                                    value="{{ $item->penulis }}" required>
                                                            </div>

                                                            <div class="mb-3">
                                                                <label for="gambar_artikel{{ $item->id }}"
                                                                    class="form-label fw-semibold">Gambar Artikel
                                                                    (Opsional)</label>
                                                                <input class="form-control shadow-sm" type="file"
                                                                    id="gambar_artikel{{ $item->id }}"
                                                                    name="gambar_artikel" accept="image/*">
                                                                @if ($item->gambar_artikel)
                                                                    <div class="mt-2">
                                                                        <img src="{{ asset('storage/' . $item->gambar_artikel) }}"
                                                                            alt="Gambar Lama"
                                                                            class="img-fluid rounded shadow-sm"
                                                                            style="max-height: 150px;">
                                                                    </div>
                                                                @endif
                                                            </div>
                                                        </div>

                                                        <!-- Kolom Kanan -->
                                                        <div class="col-md-6">
                                                            <div class="mb-3 h-100 d-flex flex-column">
                                                                <label for="isi{{ $item->id }}"
                                                                    class="form-label fw-semibold">Isi Artikel</label>
                                                                <textarea class="form-control shadow-sm flex-grow-1" id="isi{{ $item->id }}" name="isi" rows="10"
                                                                    required>{{ $item->isi }}</textarea>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                                <div class="modal-footer bg-light rounded-bottom-4">
                                                    <button type="button" class="btn btn-secondary rounded-pill px-4"
                                                        data-bs-dismiss="modal">Batal</button>
                                                    <button type="button" class="btn btn-primary rounded-pill px-4"
                                                        onclick="confirmEdit('{{ $item->id }}')">
                                                        <i class="bi bi-save me-1"></i> Simpan Perubahan
                                                    </button>
                                                </div>
                                            </form>
                                        </div>
                                    </div>
                                </div>
                            @empty
                                <tr>
                                    <td colspan="5" class="text-center text-muted">Belum ada artikel yang tersedia.
                                    </td>
                                </tr>
                            @endforelse
                        </tbody>
                    </table>
                </div>
                <div class="d-flex justify-content-between align-items-center mt-3 flex-wrap gap-2 px-4 mb-5">
                    <div class="small text-muted">
                        Showing
                        <strong>{{ $artikel->firstItem() ?? 0 }}</strong>
                        to
                        <strong>{{ $artikel->lastItem() ?? 0 }}</strong>
                        of
                        <strong>{{ $artikel->total() }}</strong>
                        entries
                    </div>

                    <nav>
                        <ul class="pagination mb-0">
                            {{-- Previous Page Link --}}
                            @if ($artikel->onFirstPage())
                                <li class="page-item disabled"><span class="page-link">&laquo;</span></li>
                            @else
                                <li class="page-item"><a class="page-link" href="{{ $artikel->previousPageUrl() }}"
                                        rel="prev">&laquo;</a></li>
                            @endif

                            {{-- Pagination Elements --}}
                            @foreach ($artikel->getUrlRange(1, $artikel->lastPage()) as $page => $url)
                                @if ($page == $artikel->currentPage())
                                    <li class="page-item active"><span class="page-link">{{ $page }}</span>
                                    </li>
                                @elseif ($page == 1 || $page == $artikel->lastPage() || abs($page - $artikel->currentPage()) <= 1)
                                    <li class="page-item"><a class="page-link"
                                            href="{{ $url }}">{{ $page }}</a></li>
                                @elseif ($page == 2 || $page == $artikel->lastPage() - 1)
                                    <li class="page-item disabled"><span class="page-link">...</span></li>
                                @endif
                            @endforeach

                            {{-- Next Page Link --}}
                            @if ($artikel->hasMorePages())
                                <li class="page-item"><a class="page-link" href="{{ $artikel->nextPageUrl() }}"
                                        rel="next">&raquo;</a></li>
                            @else
                                <li class="page-item disabled"><span class="page-link">&raquo;</span></li>
                            @endif
                        </ul>
                    </nav>
                </div>
            </main>
        </div>
    </div>


    <!-- Modal Tambah Artikel -->
    <div class="modal fade" id="modalTambah" tabindex="-1" aria-labelledby="modalTambahLabel" aria-hidden="true">
        <div class="modal-dialog modal-lg">
            <div class="modal-content rounded-4 border-0 shadow">
                <form id="tambahForm" action="{{ route('admin.artikel.store') }}" method="POST"
                    enctype="multipart/form-data">
                    @csrf
                    <div class="modal-header bg-primary text-white rounded-top-4">
                        <h5 class="modal-title fw-semibold" id="modalTambahLabel">Tambah Artikel</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"
                            aria-label="Tutup"></button>
                    </div>
                    <div class="modal-body">

                        <div class="mb-3">
                            <label for="judul" class="form-label fw-semibold">Judul Artikel</label>
                            <input type="text" class="form-control shadow-sm" id="judul" name="judul"
                                required>
                        </div>

                        <div class="mb-3">
                            <label for="penulis" class="form-label fw-semibold">Penulis Artikel</label>
                            <input type="text" class="form-control shadow-sm" id="penulis" name="penulis"
                                required>
                        </div>

                        <div class="mb-3">
                            <label for="isi" class="form-label fw-semibold">Isi Artikel</label>
                            <textarea class="form-control shadow-sm" id="isi" name="isi" rows="6" required></textarea>
                        </div>

                        <div class="mb-3">
                            <label for="gambar_artikel" class="form-label fw-semibold">Gambar Artikel</label>
                            <input class="form-control shadow-sm" type="file" id="gambar_artikel"
                                name="gambar_artikel" accept="image/*">
                        </div>

                    </div>
                    <div class="modal-footer bg-light rounded-bottom-4">
                        <button type="button" class="btn btn-secondary rounded-pill px-4"
                            data-bs-dismiss="modal">Batal</button>
                        <button type="button" class="btn btn-primary rounded-pill px-4" onclick="confirmTambah()">
                            <i class="bi bi-save me-1"></i> Simpan Artikel
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>


    <!-- Modal Lihat Artikel -->
    <div class="modal fade" id="lihatArtikelModal" tabindex="-1" aria-labelledby="lihatArtikelModalLabel"
        aria-hidden="true">
        <div class="modal-dialog modal-xl modal-dialog-scrollable">
            <div class="modal-content border-0 rounded-4 modal-article-content">
                <div class="modal-header border-0">
                    <h5 class="modal-title fw-semibold" id="lihatArtikelModalLabel">Detail Artikel</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Tutup"></button>
                </div>
                <div class="modal-body px-4 pb-4">
                    <div id="detailGambar" class="mb-4 text-center">
                        <img src="" alt="Gambar Artikel" class="img-fluid">
                    </div>
                    <h3 id="detailJudul" class="text-center"></h3>
                    <div class="text-muted mb-4 text-center">
                        <small>Ditulis oleh <span id="detailPenulis"></span> pada <span
                                id="detailTanggal"></span></small>
                    </div>
                    <div id="detailIsi" class="article-content"></div>
                </div>
            </div>
        </div>
    </div>



    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
    <script>
        @if (session('success'))
            Swal.fire({
                icon: 'success',
                title: 'Berhasil!',
                text: '{{ session('success') }}',
                timer: 3000,
                showConfirmButton: false
            });
        @endif

        @if (session('error'))
            Swal.fire({
                icon: 'error',
                title: 'Gagal!',
                text: '{{ session('error') }}',
                timer: 3000,
                showConfirmButton: false
            });
        @endif

        @if (session('message'))
            Swal.fire({
                icon: '{{ session('alert-type') == 'warning' ? 'warning' : 'info' }}',
                title: '{{ ucfirst(session('alert-type') ?? 'Info') }}',
                text: '{{ session('message') }}',
                timer: 3000,
                showConfirmButton: false
            });
        @endif

        function confirmDelete(id) {
            Swal.fire({
                title: 'Hapus Artikel?',
                text: "Data artikel yang dihapus tidak dapat dikembalikan.",
                icon: 'warning',
                showCancelButton: true,
                confirmButtonText: 'Ya, Hapus!',
                cancelButtonText: 'Batal',
                confirmButtonColor: '#d33',
                cancelButtonColor: '#6c757d'
            }).then((result) => {
                if (result.isConfirmed) {
                    const form = document.createElement('form');
                    form.method = 'POST';
                    form.action = `/admin/delete-artikel/${id}`;
                    form.innerHTML = `@csrf`;
                    document.body.appendChild(form);
                    form.submit();
                }
            });
        }


        function confirmTambah() {
            Swal.fire({
                title: 'Simpan Artikel?',
                text: "Pastikan semua informasi sudah lengkap dan benar.",
                icon: 'question',
                showCancelButton: true,
                confirmButtonColor: '#198754',
                cancelButtonColor: '#6c757d',
                confirmButtonText: 'Ya, Simpan',
                cancelButtonText: 'Batal'
            }).then((result) => {
                if (result.isConfirmed) {
                    document.getElementById('tambahForm').submit();
                }
            });
        }

        function confirmEdit(id) {
            Swal.fire({
                title: 'Perbarui Artikel?',
                text: "Pastikan semua data sudah benar.",
                icon: 'question',
                showCancelButton: true,
                confirmButtonColor: '#0d6efd',
                cancelButtonColor: '#6c757d',
                confirmButtonText: 'Ya, perbarui!',
                cancelButtonText: 'Batal'
            }).then((result) => {
                if (result.isConfirmed) {
                    document.getElementById('editForm' + id).submit();
                }
            });
        }
    </script>
    <script>
        function confirmLogout() {
            event.preventDefault();
            Swal.fire({
                title: 'Yakin ingin logout?',
                text: "Anda akan keluar dari akun ini.",
                icon: 'warning',
                showCancelButton: true,
                confirmButtonColor: '#d33',
                cancelButtonColor: '#6c757d',
                confirmButtonText: 'Ya, logout!',
                cancelButtonText: 'Batal'
            }).then((result) => {
                if (result.isConfirmed) {
                    document.getElementById('logout-form').submit();
                }
            });
        }
    </script>
    <script>
        function openDetailArtikel(btn) {
            const judul = btn.getAttribute('data-judul');
            const penulis = btn.getAttribute('data-penulis');
            const tanggal = btn.getAttribute('data-tanggal');
            const isi = btn.getAttribute('data-isi');
            const gambar = btn.getAttribute('data-gambar');

            document.getElementById('detailJudul').textContent = judul;
            document.getElementById('detailPenulis').textContent = penulis;
            document.getElementById('detailTanggal').textContent = tanggal;
            document.getElementById('detailIsi').innerHTML = decodeHTMLEntities(isi);
            document.getElementById('detailGambar').querySelector('img').src = gambar;

            const modal = new bootstrap.Modal(document.getElementById('lihatArtikelModal'));
            modal.show();
        }

        function decodeHTMLEntities(text) {
            const textarea = document.createElement('textarea');
            textarea.innerHTML = text;
            return textarea.value;
        }
    </script>

</body>

</html>
