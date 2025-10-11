<?php

namespace App\Http\Controllers;

use Carbon\Carbon;
use App\Models\User;
use App\Models\Loker;
use App\Models\Artikel;
use App\Models\Pelatihan;
use App\Models\DaftarLoker;
use App\Models\Sertifikasi;
use App\Models\MitraProfile;
use Illuminate\Http\Request;
use App\Models\DaftarPelatihan;
use App\Models\DaftarSertifikasi;
use App\Models\AlumniSiswaProfile;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Storage;

class AdminController extends Controller
{
    public function index()
    {
        $user = Auth::user();
        $totalSertifikasi = Sertifikasi::count();
        $totalPelatihan = Pelatihan::count();
        $totalLoker = Loker::count();
        $jumlahSekolah = AlumniSiswaProfile::whereNotNull('asal_sekolah')->distinct()->count('asal_sekolah');
        $jumlahPeserta = AlumniSiswaProfile::count();
        $jumlahMitra = MitraProfile::count();
        $mitras = MitraProfile::with('user')->latest()->get();

        return view('admin.dashboard', [
            'title' => 'Dashboard',
            'user' => $user,
            'totalSertifikasi' => $totalSertifikasi,
            'totalPelatihan' => $totalPelatihan,
            'totalLoker' => $totalLoker,
            'jumlahSekolah' => $jumlahSekolah,
            'jumlahPeserta' => $jumlahPeserta,
            'jumlahMitra' => $jumlahMitra,
            'mitras' => $mitras,
        ]);
    }

    public function artikel(Request $request)
    {
        $query = Artikel::query();

        // Filter: Nama Penulis
        if ($request->filled('search')) {
            $query->where('penulis', 'like', '%' . $request->search . '%');
        }

        // Filter: Bulan Dibuat
        if ($request->filled('bulan_dibuat')) {
            $query->whereMonth('created_at', $request->bulan_dibuat);
        }

        // Filter: Tahun Dibuat
        if ($request->filled('tahun_dibuat')) {
            $query->whereYear('created_at', $request->tahun_dibuat);
        }

        $artikel = $query->orderBy('created_at', 'desc')->paginate(10)->withQueryString();

        return view('admin.artikel', [
            'title' => 'Artikel',
            'artikel' => $artikel,
        ]);
    }


    public function sertifikasi(Request $request)
    {
        $query = Sertifikasi::with(['user.mitraProfile', 'daftarSertifikasis.user'])
            ->withCount('daftarSertifikasis')
            ->with([
                'user.mitraProfile',
                'daftarSertifikasis.user',
                'daftarSertifikasis' => function ($q) {
                    $q->select(
                        'id',
                        'user_id',
                        'sertifikasi_id',
                        'email',
                        'no_hp',
                        'nama_lengkap',
                        'asal_sekolah',
                        'jurusan',
                        'jenis_kelamin',
                        'tanggal_lahir',
                        'status_saat_ini',
                        'bukti_transfer'
                    );
                }
            ])
            ->orderBy('created_at', 'desc');

        if ($request->filled('mitra')) {
            $query->whereHas('user.mitraProfile', function ($q) use ($request) {
                $q->where('nama_instansi', 'like', '%' . $request->mitra . '%');
            });
        }

        if ($request->filled('judul')) {
            $query->where('judul_sertifikasi', 'like', '%' . $request->judul . '%');
        }

        if ($request->filled('kota')) {
            $query->where('kota', 'like', '%' . $request->kota . '%');
        }

        if ($request->filled('status')) {
            $query->where('status', $request->status);
        }

        if ($request->filled('bulan_mulai')) {
            $query->whereMonth('tanggal_mulai', $request->bulan_mulai);
        }

        if ($request->filled('tahun_mulai')) {
            $query->whereYear('tanggal_mulai', $request->tahun_mulai);
        }

        if ($request->filled('status_selesai')) {
            if ($request->status_selesai == 'selesai') {
                $query->whereDate('tanggal_selesai', '<', Carbon::today());
            } elseif ($request->status_selesai == 'belum') {
                $query->whereDate('tanggal_selesai', '>=', Carbon::today());
            }
        }

        $sertifikasis = $query->paginate(10)->through(function ($s) {
            $pesertas = collect($s->daftarSertifikasis);
            $s->jumlah_asal_sekolah = collect($s->daftarSertifikasis)->pluck('asal_sekolah')->unique()->count();
            $s->jumlah_jurusan = collect($s->daftarSertifikasis)->pluck('jurusan')->unique()->count();
            $s->jumlah_kuliah = $pesertas->where('status_saat_ini', 'Kuliah')->count();
            $s->jumlah_bekerja_dan_usaha = $pesertas->whereIn('status_saat_ini', ['Bekerja', 'Wirausaha'])->count();
            $s->jumlah_tidak_bekerja = $pesertas->where('status_saat_ini', 'Tidak Bekerja')->count();
            return $s;
        });

        $totalSertifikasi = Sertifikasi::count();
        $totalAlumni = User::where('role', 'alumnisiswa')->count();
        $totalSertifikasi = Sertifikasi::count();

        $sertifikasiTerdaftar = DaftarSertifikasi::distinct('sertifikasi_id')->count('sertifikasi_id');

        // Hitung persentase sertifikasi yang telah diikuti oleh alumni
        $persentaseSertifikasiTerdaftar = $totalSertifikasi > 0
            ? round(($sertifikasiTerdaftar / $totalSertifikasi) * 100, 2)
            : 0;
        $totalSertifikasiSelesai = Sertifikasi::whereDate('tanggal_selesai', '<', now())->count();

        return view('admin.sertifikasi', [
            'title' => 'Sertifikasi',
            'sertifikasis' => $sertifikasis,
            'totalSertifikasi' => $totalSertifikasi,
            'totalUser' => $totalAlumni,
            'userTerdaftar' => $sertifikasiTerdaftar,
            'persentaseUserTerdaftar' => $persentaseSertifikasiTerdaftar,
            'totalSertifikasiSelesai' => $totalSertifikasiSelesai
        ]);
    }


    /**
     * Menangani aksi store/update/delete sertifikasi.
     */
    public function handleSertifikasi(Request $request)
    {
        $action = $request->input('action');

        // ===== STORE =====
        if ($action === 'store') {
            $validated = $request->validate([
                'judul_sertifikasi' => 'required|string|max:255',
                'deskripsi' => 'required|string',
                'tanggal_mulai' => 'required|date',
                'tanggal_selesai' => 'required|date|after_or_equal:tanggal_mulai',
                'biaya' => 'nullable|numeric',
                'kota' => 'required|string|max:100',
                'tempat' => 'required|string|max:255',
                'status' => 'required|string|in:Aktif,Nonaktif',
                'foto_sertifikasi' => 'required|image|mimes:jpg,jpeg,png|max:2048',
                // tidak perlu validasi user_id/mitra karena otomatis dari login
            ]);

            $sertifikasi = new Sertifikasi();
            $sertifikasi->user_id = auth()->id(); // asumsi: si mitra sedang login
            $sertifikasi->judul_sertifikasi = $validated['judul_sertifikasi'];
            $sertifikasi->deskripsi = $validated['deskripsi'];
            $sertifikasi->tanggal_mulai = $validated['tanggal_mulai'];
            $sertifikasi->tanggal_selesai = $validated['tanggal_selesai'];
            $sertifikasi->biaya = $validated['biaya'];
            $sertifikasi->kota = $validated['kota'];
            $sertifikasi->tempat = $validated['tempat'];
            $sertifikasi->status = $validated['status'];
            // simpan foto
            $sertifikasi->foto_sertifikasi = $request->file('foto_sertifikasi')->store('sertifikasis', 'public');
            $sertifikasi->save();

            return redirect()->route('admin.sertifikasi')->with('success', 'Sertifikasi berhasil ditambahkan.');
        }

        // ===== UPDATE =====
        if ($action === 'update') {
            $validated = $request->validate([
                'id' => 'required|exists:sertifikasis,id',
                'judul_sertifikasi' => 'required|string|max:255',
                'deskripsi' => 'required|string',
                'tanggal_mulai' => 'required|date',
                'tanggal_selesai' => 'required|date|after_or_equal:tanggal_mulai',
                'biaya' => 'nullable|numeric',
                'kota' => 'required|string|max:100',
                'tempat' => 'required|string|max:255',
                'status' => 'required|string|in:Aktif,Nonaktif',
                'foto_sertifikasi' => 'nullable|image|mimes:jpg,jpeg,png|max:2048',
            ]);

            $sertifikasi = Sertifikasi::findOrFail($validated['id']);
            $sertifikasi->judul_sertifikasi = $validated['judul_sertifikasi'];
            $sertifikasi->deskripsi = $validated['deskripsi'];
            $sertifikasi->tanggal_mulai = $validated['tanggal_mulai'];
            $sertifikasi->tanggal_selesai = $validated['tanggal_selesai'];
            $sertifikasi->biaya = $validated['biaya'];
            $sertifikasi->kota = $validated['kota'];
            $sertifikasi->tempat = $validated['tempat'];
            $sertifikasi->status = $validated['status'];

            // Jika ada foto baru, hapus foto lama & simpan yang baru
            if ($request->hasFile('foto_sertifikasi')) {
                if ($sertifikasi->foto_sertifikasi) {
                    Storage::disk('public')->delete($sertifikasi->foto_sertifikasi);
                }
                $sertifikasi->foto_sertifikasi = $request->file('foto_sertifikasi')->store('sertifikasis', 'public');
            }

            $sertifikasi->save();
            return redirect()->route('admin.sertifikasi')->with('success', 'Sertifikasi berhasil diperbarui.');
        }

        // ===== DELETE =====
        if ($action === 'delete') {
            $request->validate([
                'id' => 'required|exists:sertifikasis,id',
            ]);

            $sertifikasi = Sertifikasi::findOrFail($request->id);
            if ($sertifikasi->foto_sertifikasi) {
                Storage::disk('public')->delete($sertifikasi->foto_sertifikasi);
            }
            $sertifikasi->delete();
            return redirect()->route('admin.sertifikasi')->with('success', 'Sertifikasi berhasil dihapus.');
        }

        // Jika tidak ada action yang cocok, kembali saja
        return redirect()->route('admin.sertifikasi');
    }

    public function loker(Request $request)
    {
        $lokersQuery = Loker::with(['user.mitraProfile', 'daftarLoker.user'])
            ->withCount('daftarLoker')
            ->with([
                'user.mitraProfile',
                'daftarLoker.user',
                'daftarLoker' => function ($q) {
                    $q->select(
                        'id',
                        'user_id',
                        'loker_id',
                        'email',
                        'no_hp',
                        'nama_lengkap',
                        'asal_sekolah',
                        'jurusan',
                        'jenis_kelamin',
                        'tanggal_lahir',
                        'status_saat_ini',
                        'cv'
                    );
                }
            ])
            ->orderBy('created_at', 'desc');

        if ($request->filled('mitra')) {
            $lokersQuery->whereHas('user.mitraProfile', function ($q) use ($request) {
                $q->where('nama_instansi', 'like', '%' . $request->mitra . '%');
            });
        }

        if ($request->filled('nama_perusahaan')) {
            $lokersQuery->where('nama_perusahaan', 'like', '%' . $request->nama_perusahaan . '%');
        }

        if ($request->filled('lokasi')) {
            $lokersQuery->where('lokasi', 'like', '%' . $request->lokasi . '%');
        }

        if ($request->filled('tipe')) {
            $lokersQuery->where('tipe', $request->tipe);
        }

        if ($request->filled('status')) {
            $lokersQuery->where('status', $request->status);
        }


        // === PAGINATION 10 PER HALAMAN ===
        $lokers = $lokersQuery->paginate(10)->through(function ($p) {
            $pesertas = collect($p->daftarLoker);
            $p->jumlah_asal_sekolah = collect($p->daftarLoker)->pluck('asal_sekolah')->unique()->count();
            $p->jumlah_jurusan = collect($p->daftarLoker)->pluck('jurusan')->unique()->count();
            $p->jumlah_kuliah = $pesertas->where('status_saat_ini', 'Kuliah')->count();
            $p->jumlah_bekerja_dan_usaha = $pesertas->whereIn('status_saat_ini', ['Bekerja', 'Wirausaha'])->count();
            $p->jumlah_tidak_bekerja = $pesertas->where('status_saat_ini', 'Tidak Bekerja')->count();
            return $p;
        });

        // === METRICS ===
        $totalLoker = Loker::count();
        $totalUser = User::count();
        $lokerTerdaftar = DaftarLoker::distinct('loker_id')->count('loker_id');

        // Hitung persentase sertifikasi yang telah diikuti oleh alumni
        $persentaseLokerTerdaftar = $totalLoker > 0
            ? round(($lokerTerdaftar / $totalLoker) * 100, 2)
            : 0;

        return view('admin.loker', [
            'title' => 'Loker',
            'lokers' => $lokers,
            'totalLoker' => $totalLoker,
            'totalUser' => $totalUser,
            'lokerTerdaftar' => $lokerTerdaftar,
            'persentaseLokerTerdaftar' => $persentaseLokerTerdaftar,
        ]);
    }

    // Menangani aksi store/update/delete
    public function handleLoker(Request $request)
    {
        $action = $request->input('action');

        switch ($action) {
            case 'store':
                return $this->storeLoker($request);
            case 'update':
                return $this->updateLoker($request);
            case 'delete':
                return $this->deleteLoker($request);
            default:
                abort(400, 'Aksi tidak dikenali');
        }
    }

    // Tambah loker baru
    protected function storeLoker(Request $request)
    {
        $data = $request->validate([
            'nama_perusahaan' => 'required|string',
            'posisi' => 'required|string',
            'lokasi' => 'required|string',
            'tipe' => 'required|string',
            'pendidikan' => 'required|string',
            'gaji' => 'nullable|string',
            'deskripsi' => 'required|string',
            'gambar' => 'required|image|max:2048',
        ]);

        $data['gambar'] = $request->file('gambar')->store('loker', 'public');
        $data['user_id'] = auth()->id(); // atau bisa ditentukan id mitra tertentu

        Loker::create($data);

        return redirect()->route('admin.loker')->with('success', 'Loker berhasil ditambahkan.');
    }

    // Update loker
    protected function updateLoker(Request $request)
    {
        $loker = Loker::findOrFail($request->id);

        $data = $request->validate([
            'nama_perusahaan' => 'required|string',
            'posisi' => 'required|string',
            'lokasi' => 'required|string',
            'tipe' => 'required|string',
            'pendidikan' => 'required|string',
            'gaji' => 'nullable|string',
            'deskripsi' => 'required|string',
            'gambar' => 'nullable|image|max:2048',
        ]);

        if ($request->hasFile('gambar')) {
            // Hapus gambar lama
            if ($loker->gambar && Storage::disk('public')->exists($loker->gambar)) {
                Storage::disk('public')->delete($loker->gambar);
            }
            $data['gambar'] = $request->file('gambar')->store('loker', 'public');
        }

        $loker->update($data);

        return redirect()->route('admin.loker')->with('success', 'Loker berhasil diperbarui.');
    }

    // Hapus loker
    protected function deleteLoker(Request $request)
    {
        $loker = Loker::findOrFail($request->id);

        if ($loker->gambar && Storage::disk('public')->exists($loker->gambar)) {
            Storage::disk('public')->delete($loker->gambar);
        }

        $loker->delete();

        return redirect()->route('admin.loker')->with('success', 'Loker berhasil dihapus.');
    }


    public function pelatihan(Request $request)
    {
        if ($request->isMethod('post')) {
            $action = $request->input('action');

            if ($action === 'store') {
                $validated = $request->validate([
                    'nama_pelatihan' => 'required',
                    'tempat_pelatihan' => 'required',
                    'tanggal_mulai' => 'required|date',
                    'tanggal_selesai' => 'required|date|after_or_equal:tanggal_mulai',
                    'deskripsi' => 'required',
                    'biaya' => 'nullable|numeric',
                    'kota' => 'required',
                    'status' => 'required|in:Gratis,Berbayar',
                    'foto_pelatihan' => 'required|image|max:2048',
                ]);

                $validated['user_id'] = auth()->id();
                $validated['foto_pelatihan'] = $request->file('foto_pelatihan')->store('pelatihan');

                Pelatihan::create($validated);

                return redirect()->back()->with('success', 'Pelatihan berhasil ditambahkan.');
            }

            if ($action === 'update') {
                $pelatihan = Pelatihan::findOrFail($request->input('id'));

                $data = $request->except(['action', 'id', 'foto_pelatihan']);

                if ($request->hasFile('foto_pelatihan')) {
                    // Hapus foto lama (opsional, aktifkan jika mau)
                    // if ($pelatihan->foto_pelatihan) {
                    //     Storage::delete($pelatihan->foto_pelatihan);
                    // }
                    $data['foto_pelatihan'] = $request->file('foto_pelatihan')->store('pelatihan');
                }

                $pelatihan->update($data);

                return redirect()->back()->with('success', 'Pelatihan berhasil diupdate.');
            }

            if ($action === 'delete') {
                Pelatihan::destroy($request->input('id'));
                return redirect()->back()->with('success', 'Pelatihan berhasil dihapus.');
            }
        }

        // === QUERY with FILTER ===
        $pelatihansQuery = Pelatihan::with(['user.mitraProfile', 'daftarPelatihan.user'])
            ->withCount('daftarPelatihan')
            ->with([
                'user.mitraProfile',
                'daftarPelatihan.user',
                'daftarPelatihan' => function ($q) {
                    $q->select(
                        'id',
                        'user_id',
                        'pelatihan_id',
                        'email',
                        'no_hp',
                        'nama_lengkap',
                        'asal_sekolah',
                        'jurusan',
                        'jenis_kelamin',
                        'tanggal_lahir',
                        'status_saat_ini',
                        'bukti_transfer'
                    );
                }
            ])
            ->orderBy('created_at', 'desc');

        if ($request->filled('search')) {
            $pelatihansQuery->where('nama_pelatihan', 'like', '%' . $request->search . '%');
        }

        if ($request->filled('mitra')) {
            $pelatihansQuery->whereHas('user.mitraProfile', function ($q) use ($request) {
                $q->where('nama_instansi', 'like', '%' . $request->mitra . '%');
            });
        }

        if ($request->filled('kota')) {
            $pelatihansQuery->where('kota', 'like', '%' . $request->kota . '%');
        }

        if ($request->filled('status')) {
            $pelatihansQuery->where('status', $request->status);
        }

        if ($request->filled('bulan_mulai')) {
            $pelatihansQuery->whereMonth('tanggal_mulai', $request->bulan_mulai);
        }

        if ($request->filled('tahun_mulai')) {
            $pelatihansQuery->whereYear('tanggal_mulai', $request->tahun_mulai);
        }

        if ($request->filled('status_selesai')) {
            if ($request->status_selesai == 'selesai') {
                $pelatihansQuery->whereDate('tanggal_selesai', '<', Carbon::today());
            } elseif ($request->status_selesai == 'belum') {
                $pelatihansQuery->whereDate('tanggal_selesai', '>=', Carbon::today());
            }
        }


        // === PAGINATION 10 PER HALAMAN ===
        $pelatihans = $pelatihansQuery->paginate(10)->through(function ($p) {
            $pesertas = collect($p->daftarPelatihan);
            $p->jumlah_asal_sekolah = collect($p->daftarPelatihan)->pluck('asal_sekolah')->unique()->count();
            $p->jumlah_jurusan = collect($p->daftarPelatihan)->pluck('jurusan')->unique()->count();
            $p->jumlah_kuliah = $pesertas->where('status_saat_ini', 'Kuliah')->count();
            $p->jumlah_bekerja_dan_usaha = $pesertas->whereIn('status_saat_ini', ['Bekerja', 'Wirausaha'])->count();
            $p->jumlah_tidak_bekerja = $pesertas->where('status_saat_ini', 'Tidak Bekerja')->count();
            return $p;
        });

        // === METRICS ===
        $totalPelatihan = Pelatihan::count();
        $totalUser = User::count();
        $pelatihanTerdaftar = DaftarPelatihan::distinct('pelatihan_id')->count('pelatihan_id');

        // Hitung persentase sertifikasi yang telah diikuti oleh alumni
        $persentasePelatihanTerdaftar = $totalPelatihan > 0
            ? round(($pelatihanTerdaftar / $totalPelatihan) * 100, 2)
            : 0;
        $totalPelatihanSelesai = Pelatihan::whereDate('tanggal_selesai', '<', now())->count();

        return view('admin.pelatihan', [
            'title' => 'Pelatihan',
            'pelatihans' => $pelatihans,
            'totalPelatihan' => $totalPelatihan,
            'totalUser' => $totalUser,
            'pelatihanTerdaftar' => $pelatihanTerdaftar,
            'persentasePelatihanTerdaftar' => $persentasePelatihanTerdaftar,
            'totalPelatihanSelesai' => $totalPelatihanSelesai
        ]);
    }

    public function handlePelatihan(Request $request)
    {
        $action = $request->input('action');

        switch ($action) {
            case 'store':
                return $this->storePelatihan($request);
            case 'update':
                return $this->updatePelatihan($request);
            case 'delete':
                return $this->deletePelatihan($request);
            default:
                abort(400, 'Aksi tidak dikenali');
        }
    }

    // Tambah loker baru
    protected function storePelatihan(Request $request)
    {
        $data = $request->validate([
            'nama_perusahaan' => 'required|string',
            'posisi' => 'required|string',
            'lokasi' => 'required|string',
            'tipe' => 'required|string',
            'pendidikan' => 'required|string',
            'gaji' => 'nullable|string',
            'deskripsi' => 'required|string',
            'gambar' => 'required|image|max:2048',
        ]);

        $data['gambar'] = $request->file('gambar')->store('loker', 'public');
        $data['user_id'] = auth()->id(); // atau bisa ditentukan id mitra tertentu

        Loker::create($data);

        return redirect()->route('admin.loker')->with('success', 'Loker berhasil ditambahkan.');
    }

    // Update loker
    protected function updatePelatihan(Request $request)
    {
        $loker = Loker::findOrFail($request->id);

        $data = $request->validate([
            'nama_perusahaan' => 'required|string',
            'posisi' => 'required|string',
            'lokasi' => 'required|string',
            'tipe' => 'required|string',
            'pendidikan' => 'required|string',
            'gaji' => 'nullable|string',
            'deskripsi' => 'required|string',
            'gambar' => 'nullable|image|max:2048',
        ]);

        if ($request->hasFile('gambar')) {
            // Hapus gambar lama
            if ($loker->gambar && Storage::disk('public')->exists($loker->gambar)) {
                Storage::disk('public')->delete($loker->gambar);
            }
            $data['gambar'] = $request->file('gambar')->store('loker', 'public');
        }

        $loker->update($data);

        return redirect()->route('admin.loker')->with('success', 'Loker berhasil diperbarui.');
    }

    // Hapus loker
    protected function deletePelatihan(Request $request)
    {
        $pelatihan = Pelatihan::findOrFail($request->id);

        if ($pelatihan->foto_pelatihan && Storage::disk('public')->exists($pelatihan->foto_pelatihan)) {
            Storage::disk('public')->delete($pelatihan->foto_pelatihan);
        }

        $pelatihan->delete();

        return redirect()->route('admin.pelatihan')->with('success', 'Pelatihan berhasil dihapus.');
    }

    public function users(Request $request)
    {
        // Mulai query dengan Eloquent, gunakan eager loading 'user' langsung di sini
        $profilesQuery = AlumniSiswaProfile::with('user');

        // Filter pencarian
        if ($request->filled('search')) {
            $profilesQuery->where('nama_lengkap', 'like', '%' . $request->search . '%');
        }
        if ($request->filled('asal_sekolah')) {
            $profilesQuery->where('asal_sekolah', 'like', '%' . $request->asal_sekolah . '%');
        }
        if ($request->filled('jurusan')) {
            $profilesQuery->where('jurusan_sekolah', 'like', '%' . $request->jurusan . '%');
        }
        if ($request->filled('tahun_kelulusan')) {
            $profilesQuery->where('tahun_kelulusan', $request->tahun_kelulusan);
        }
        if ($request->filled('nama_universitas')) {
            $profilesQuery->where('nama_universitas', 'like', '%' . $request->nama_universitas . '%');
        }
        if ($request->filled('status')) {
            $profilesQuery->where('status_saat_ini', $request->status);
        }
        if ($request->filled('tanggal_mulai') && $request->filled('tanggal_selesai')) {
            $profilesQuery->whereBetween('created_at', [
                $request->tanggal_mulai . ' 00:00:00',
                $request->tanggal_selesai . ' 23:59:59'
            ]);
        } elseif ($request->filled('tanggal_mulai')) {
            $profilesQuery->where('created_at', '>=', $request->tanggal_mulai . ' 00:00:00');
        } elseif ($request->filled('tanggal_selesai')) {
            $profilesQuery->where('created_at', '<=', $request->tanggal_selesai . ' 23:59:59');
        }

        // Ambil hasil query setelah filter selesai
        $profiles = $profilesQuery->paginate(10)->withQueryString();

        // Statistik
        $totalAlumni = AlumniSiswaProfile::count();
        $totalSekolah = AlumniSiswaProfile::distinct('asal_sekolah')->count('asal_sekolah');
        $totalJurusan = AlumniSiswaProfile::distinct('jurusan_sekolah')->count('jurusan_sekolah');

        return view('admin.usersAlumni', [
            'title' => 'Data Users Alumni',
            'profiles' => $profiles,
            'totalAlumni' => $totalAlumni,
            'totalSekolah' => $totalSekolah,
            'totalJurusan' => $totalJurusan,
        ]);
    }

    // Simpan data baru
    public function usersStore(Request $request)
    {
        $request->validate([
            'user_id' => 'required|exists:users,id',
            'nama_lengkap' => 'required',
            'nik' => 'required|unique:alumni_siswa_profiles,nik',
            'tahun_kelulusan' => 'nullable|digits:4',
            'asal_sekolah' => 'required',
            'jurusan_sekolah' => 'required',
            'jenis_kelamin' => 'required',
            'tanggal_lahir' => 'required|date',
            'npsn' => 'required',
            'provinsi' => 'required',
            'kota' => 'required',
            'alamat' => 'required',
            'status_saat_ini' => 'required',
        ]);

        AlumniSiswaProfile::create($request->all());

        return back()->with('success', 'Data user berhasil ditambahkan.');
    }


    public function usersMitra(Request $request)
    {
        // Mulai query dengan Eloquent dan eager loading relasi 'user'
        $profilesQuery = MitraProfile::with('user');

        // Filter pencarian (disesuaikan dengan field yang relevan di MitraProfile)
        if ($request->filled('search')) {
            $profilesQuery->where('nama_instansi', 'like', '%' . $request->search . '%');
        }
        if ($request->filled('kategori')) {
            $profilesQuery->where('kategori', 'like', '%' . $request->kategori . '%');
        }
        if ($request->filled('bidang')) {
            $profilesQuery->where('bidang_industri', 'like', '%' . $request->bidang . '%');
        }
        if ($request->filled('provinsi')) {
            $profilesQuery->where('provinsi', 'like', '%' . $request->provinsi . '%');
        }
        if ($request->filled('kota')) {
            $profilesQuery->where('kota', 'like', '%' . $request->kota . '%');
        }

        // Ambil hasil dengan pagination
        $profiles = $profilesQuery->paginate(10)->withQueryString();

        // Statistik sederhana
        $totalMitra = MitraProfile::count();
        $totalBidang = MitraProfile::distinct('bidang_industri')->count('bidang_industri');
        $totalKota = MitraProfile::distinct('kota')->count('kota');

        return view('admin.usersMitra', [
            'title' => 'Data Users Mitra',
            'profiles' => $profiles,
            'totalMitra' => $totalMitra,
            'totalBidang' => $totalBidang,
            'totalKota' => $totalKota,
        ]);
    }


    private function storeMitra(Request $request)
    {
        $request->validate([
            'nama_instansi' => 'required|string',
            'penanggung_jawab' => 'required|string',
            'bidang_industri' => 'required|string',
            'kategori' => 'required|string',
            'provinsi' => 'required|string',
            'kota' => 'required|string',
            'alamat' => 'required|string',
            'email' => 'required|email|unique:users,email',
        ]);

        DB::transaction(function () use ($request) {
            $user = User::create([
                'name' => $request->penanggung_jawab,
                'email' => $request->email,
                'password' => bcrypt('password'),
                'role' => 'mitra',
            ]);

            MitraProfile::create([
                'user_id' => $user->id,
                'nama_instansi' => $request->nama_instansi,
                'penanggung_jawab' => $request->penanggung_jawab,
                'bidang_industri' => $request->bidang_industri,
                'kategori' => $request->kategori,
                'provinsi' => $request->provinsi,
                'kota' => $request->kota,
                'alamat' => $request->alamat,
            ]);
        });

        return redirect()->back()->with('success', 'Data mitra berhasil ditambahkan.');
    }

    private function updateMitra(Request $request)
    {
        $request->validate([
            'id' => 'required|exists:mitra_profiles,id',
        ]);

        $profile = MitraProfile::findOrFail($request->id);
        $profile->update($request->except('_token', 'action', 'id'));

        return redirect()->back()->with('success', 'Data mitra berhasil diperbarui.');
    }

    private function deleteMitra(Request $request)
    {
        $profile = MitraProfile::findOrFail($request->id);
        $profile->delete();

        return redirect()->back()->with('success', 'Data mitra berhasil dihapus.');
    }
}
